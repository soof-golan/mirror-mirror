import asyncio
import base64
import threading
from io import BytesIO

import janus
import numpy.typing as npt
import gradio as gr
import numpy as np
from diffusers.pipelines.sana.pipeline_sana_sprint import retrieve_timesteps
from fastrtc import Stream, WebRTC, AudioVideoStreamHandler
from gradio.utils import get_space
from PIL import Image
from diffusers import (
    # SanaPAGPipeline as Pipeline,
    AutoencoderDC,
    SanaSprintPipeline as Pipeline,
    DPMSolverMultistepScheduler,
    SCMScheduler,
)
import torch

max_timesteps: float = 1.57080

intermediate_timesteps: float = 1.3
torch_dtype = torch.bfloat16
device = None
if torch.cuda.is_available():
    device = torch.device("cuda:0")

timesteps = torch.tensor([0], device=device, dtype=torch.int)
# SprintPipeline.__call__


print("Loading Model:")

PIPELINE = Pipeline.from_pretrained(
    "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
    torch_dtype=torch_dtype,
)
if device is not None:
    PIPELINE = PIPELINE.to(device)

prompt = "Oil on canvas, a portrait"

timesteps = retrieve_timesteps(
    PIPELINE.scheduler,
    2,
    device,
    None,
    sigmas=None,
    max_timesteps=max_timesteps,
    intermediate_timesteps=intermediate_timesteps,
)
in_queue = janus.Queue(1)
out_queue = janus.Queue(1)


def diffuser():
    while True:
        frame = in_queue.sync_q.get()
        if frame is None:
            break
        print("Diffusing frame")
        out = PIPELINE(
            prompt=prompt,
            num_inference_steps=2,
            output_type="np",
        ).images[0]
        out_queue.sync_q.put_nowait(out)


threading.Thread(target=diffuser, daemon=True).start()


@torch.inference_mode()
def prepare_image_prompt(
    scheduler: SCMScheduler,
    vae: AutoencoderDC,
    frame: npt.NDArray[np.uint8],
    _timestep: float = max_timesteps,
    to_device: torch.device | None = device,
    to_dtype: torch.dtype | None = torch_dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Prepare image prompt for the pipeline"""

    frame_chw = frame.transpose(2, 0, 1)
    if to_device:
        x = torch.tensor(frame_chw, device=to_device, dtype=to_dtype)
    else:
        x = torch.tensor(frame_chw, dtype=to_dtype)
    x = x.unsqueeze(0)

    init_latents = vae.encoder(x)
    # noise = torch.randn(
    #     init_latents.shape,
    #     device=init_latents.device,
    #     dtype=init_latents.dtype,
    #     generator=generator,
    # )
    # init_latents = scheduler.(init_latents, noise, timesteps)
    return init_latents


def encode_audio(data: np.ndarray) -> dict:
    """Encode Audio data to send to the server"""
    return {
        "mime_type": "audio/pcm",
        "data": base64.b64encode(data.tobytes()).decode("UTF-8"),
    }


def encode_image(data: np.ndarray) -> dict:
    with BytesIO() as output_bytes:
        pil_image = Image.fromarray(data)
        pil_image.save(output_bytes, "JPEG")
        bytes_data = output_bytes.getvalue()
    base64_str = str(base64.b64encode(bytes_data), "utf-8")
    return {"mime_type": "image/jpeg", "data": base64_str}


class AVHandler(AudioVideoStreamHandler):
    def __init__(self) -> None:
        super().__init__(
            skip_frames=True,
        )
        self.video_queue = out_queue
        self.last_frame_time = 0
        self.quit = asyncio.Event()
        self.pipeline: Pipeline = PIPELINE
        self.vae: AutoencoderDC = self.pipeline.vae
        self.scheduler: DPMSolverMultistepScheduler = self.pipeline.scheduler

    def copy(self) -> "AVHandler":
        return AVHandler()

    async def start_up(self):
        pass

    def video_receive(self, frame: np.ndarray):
        try:
            in_queue.sync_q.put_nowait(frame)
        except janus.SyncQueueFull:
            print("Input queue is full, dropping frame")
            return

        # latents = prepare_image_prompt(
        #     scheduler=self.scheduler,
        #     vae=self.vae,
        #     frame=frame,
        # )

        # out = self.pipeline.__call__(
        #     prompt=prompt,
        #     num_inference_steps=2,
        #     output_type="np",
        # ).images[0]
        #
        # self.video_queue.put_nowait(np.array(out))
        # self.video_queue.put_nowait(frame)

    def video_emit(self):
        try:
            return self.video_queue.sync_q.get(timeout=0.1)
        except janus.SyncQueueEmpty:
            pass

    async def shutdown(self) -> None:
        pass


stream = Stream(
    handler=AVHandler(),
    modality="video",
    mode="send-receive",
    time_limit=180 if get_space() else None,
    ui_args={"title": "Mirror Mirror"},
)

css = """
#video-source {max-width: 600px !important; max-height: 600 !important;}
"""

with gr.Blocks(css=css) as demo:
    with gr.Row() as row:
        with gr.Column():
            webrtc = WebRTC(
                label="Video Chat",
                modality="video",
                mode="send-receive",
                elem_id="video-source",
                icon="https://www.gstatic.com/lamda/images/gemini_favicon_f069958c85030456e93de685481c559f160ea06b.png",
                pulse_color="rgb(255, 255, 255)",
                icon_button_color="rgb(255, 255, 255)",
            )

        webrtc.stream(
            AVHandler(),
            inputs=[webrtc],
            outputs=[webrtc],
            time_limit=180 if get_space() else None,
            concurrency_limit=2 if get_space() else None,
        )

stream.ui = demo


if __name__ == "__main__":
    stream.ui.launch(server_port=7860)
