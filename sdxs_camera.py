import queue
import threading
from contextlib import suppress
from typing import NoReturn, Any

import torch
from diffusers import StableDiffusionPipeline, AutoencoderTiny, AutoencoderKL
from diffusers.models.attention_processor import AttnProcessor2_0
import cv2
from typer import Typer

app = Typer()

repo = "IDKiro/sdxs-512-dreamshaper"
seed = 42
weight_type = torch.float16  # or float32

# Load model.
pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(repo, torch_dtype=weight_type)
pipe.unet.set_attn_processor(AttnProcessor2_0())
pipe.to("cuda")

prompt = "a close-up picture of an old man standing in the rain"
g = torch.Generator(device="cuda").manual_seed(seed)
prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
    [prompt],
    num_images_per_prompt=1,
    device=pipe.device,
    do_classifier_free_guidance=False,
)


# Ensure using 1 inference step and CFG set to 0.
def gen():
    image = pipe.__call__(
        # prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        num_inference_steps=1,
        guidance_scale=0,
        # generator=g,
        output_type="latent",
    ).images[0]


def camera_loop(channels: dict[str, queue.Queue], camera_id: int) -> NoReturn:
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {camera_id}")
    while True:
        success, frame = cap.read()
        if not success:
            raise RuntimeError("Failed to capture frame")
        publish(channels, "camera", frame)


@torch.inference_mode
def encode_frame_loop(channels: dict[str, queue.Queue], pipe: StableDiffusionPipeline) -> NoReturn:
    vae: AutoencoderKL | AutoencoderTiny = pipe.vae
    while True:
        frame = recv(channels, "camera")

        image = pipe.image_processor.preprocess(frame)
        image = image.to(pipe.device, dtype=pipe.dtype)
        latents = vae.encoder.forward(image)
        publish(channels, "latents_in", latents)


@torch.inference_mode
def diffusion_loop(channels: dict[str, queue.Queue], pipe: StableDiffusionPipeline) -> NoReturn:
    prompt_embeds, negative_prompt_embeds = None, None
    while True:
        latents_in = recv(channels, "latents_in")
        prompt_embeds, negative_prompt_embeds = recv_nowait(
            channels, "prompt_embeds", default=(prompt_embeds, negative_prompt_embeds)
        )
        if prompt_embeds is None or negative_prompt_embeds is None:
            # Wait for prompt embeds to be set
            continue
        latents_out = pipe.__call__(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_inference_steps=1,
            guidance_scale=0,
            output_type="latent",
            latents=latents_in,
        )
        publish(channels, "latents_out", latents_out)


@torch.inference_mode
def decode_loop(channels: dict[str, queue.Queue], pipe: StableDiffusionPipeline) -> NoReturn:
    vae: AutoencoderKL | AutoencoderTiny = pipe.vae
    scaling_recip = 1 / vae.config.scaling_factor
    while True:
        latents = recv(channels, "latents_out")
        img = vae.decoder.forward(latents * scaling_recip)
        img = pipe.image_processor.postprocess(img, output_type="np")
        publish(channels, "image_out", img)


def display_loop(channels: dict[str, queue.Queue]) -> NoReturn:
    while True:
        frame = recv(channels, "image_out")
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            cv2.destroyAllWindows()
            raise SystemExit("Exiting display loop")


def publish(queus: dict[str, queue.Queue], channel: str, message: Any) -> None:
    if channel not in queus:
        queus[channel] = queue.Queue()
    queus[channel].put(message)


def recv(channels: dict[str, queue.Queue], channel: str) -> Any:
    if channel not in channels:
        channels[channel] = queue.Queue()
    q = channels[channel]
    obj = None
    if q.empty():
        obj = q.get()
    else:
        while not q.empty():
            obj = q.get_nowait()
    return obj


def recv_nowait(channels: dict[str, queue.Queue], channel: str, default: Any) -> Any:
    if channel not in channels:
        channels[channel] = queue.Queue()
    q = channels[channel]
    obj = default
    with suppress(queue.Empty):
        while True:
            obj = q.get_nowait()
    return obj


@app.command()
def main(camera_id: int = 0):
    channels = {}
    threading.Thread(target=camera_loop, args=(channels, camera_id), daemon=True).start()
    threading.Thread(target=encode_frame_loop, args=(channels, pipe), daemon=True).start()
    threading.Thread(target=diffusion_loop, args=(channels, pipe), daemon=True).start()
    threading.Thread(target=decode_loop, args=(channels, pipe), daemon=True).start()
    display_loop(channels)


if __name__ == "__main__":
    app()
