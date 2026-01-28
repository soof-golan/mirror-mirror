import atexit
import logging
import multiprocessing
import os
from abc import ABC, abstractmethod
from collections.abc import Iterable
from contextlib import ExitStack
from itertools import count
from typing import Any, Literal, NoReturn, TypedDict

import numpy as np
import pypeln as pl
import torch
import zmq
from diffusers import StableDiffusionImg2ImgPipeline
from tqdm import tqdm
from typer import Typer

import mirror.native_camera
from mirror.common_types import RGBFrame
from mirror.config import Config
from mirror.diffusion_configurations import get_pipeline
from mirror.encode_prompt import encode_prompt
from mirror.fused_decode_upscale import FusedDecodeUpscale, torch_to_np
from mirror.generated.messages import Frame
from mirror.native_camera import init_camera
from mirror.utils import decode_frame, encode_frame, log_timing
from mirror.zmq_utils import create_socket

app = Typer()


class Data(TypedDict, total=False):
    raw_frame: Frame
    np_frame: RGBFrame
    prompt: str
    prompt_kwargs: dict[str, Any]
    frame_tensor: torch.Tensor
    latents: torch.Tensor
    decoded_image: torch.Tensor
    final_image: RGBFrame


class Item:
    def __init__(self):
        self.data: Data = {}


class Stage(ABC):
    @abstractmethod
    def process(self, item: Item) -> Item:
        raise NotImplementedError()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.__call__ = log_timing(cls.__call__, name=f"{cls.__name__}.process")

    @torch.inference_mode()
    def __call__(self, item: Item) -> Item:
        return self.process(item)


class PromptSource(Stage, ABC):
    def __init__(
        self,
        config: Config,
        pipe: StableDiffusionImg2ImgPipeline,
        prompt_republish: zmq.Socket | None = None,
    ):
        self.prompt_sub = create_socket(zmq.SUB, config.zmq_prompt_pub_addr, "connect")
        self.state = create_socket(zmq.DEALER, config.zmq_state_addr, "connect", conflate=False)
        self.poller = zmq.Poller()
        self.poller.register(self.prompt_sub, zmq.POLLIN)

        self.current_prompt: str | None = None
        self.prompt_kwargs: dict[str, Any] = {}
        self.pipe = pipe
        self.prompt_republish = prompt_republish
        self.current_prompt = self.request_prompt_snapshot() or config.default_prompt
        self.prompt_kwargs = encode_prompt(self.pipe, self.current_prompt)

    def __del__(self):
        self.poller.unregister(self.prompt_sub)
        self.prompt_sub.close()
        self.state.close()

    def request_prompt_snapshot(self) -> str | None:
        self.state.send_multipart([b"pipeline", b"", b"get_prompt"])
        if self.state.poll(timeout=3000):
            return self.state.recv_multipart()[2].decode()
        return None

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            item = self.pop()
            if item is not None:
                return item

    @torch.inference_mode()
    def _poll_prompt(self) -> tuple[dict[zmq.Socket, Any], Item]:
        socks = dict(self.poller.poll(timeout=0))
        if self.prompt_sub in socks:
            self.current_prompt = self.prompt_sub.recv_string()
            self.prompt_kwargs = encode_prompt(self.pipe, self.current_prompt)
            if self.prompt_republish:
                self.prompt_republish.send_string(self.current_prompt)

        item = Item()
        item.data["prompt"] = self.current_prompt
        item.data["prompt_kwargs"] = self.prompt_kwargs
        return socks, item


class ZmqFrameCapture(PromptSource):
    def __init__(
        self,
        config: Config,
        pipe: StableDiffusionImg2ImgPipeline,
        prompt_republish: zmq.Socket | None = None,
    ):
        super().__init__(config, pipe, prompt_republish)
        self.frames = create_socket(zmq.SUB, config.zmq_frames_addr, "connect", conflate=True)

    def process(self, _) -> Item:
        socks, item = self._poll_prompt()
        data = Frame.parse(self.frames.recv())
        item.data["raw_frame"] = data
        return item


class NativeFrameCapture(PromptSource):
    def __init__(
        self,
        config: Config,
        pipe: StableDiffusionImg2ImgPipeline,
        cap_config: dict,
        prompt_republish: zmq.Socket | None = None,
    ):
        super().__init__(config, pipe, prompt_republish)
        self.cap_config = cap_config
        self.cap = init_camera(cap_config=cap_config)

    def __del__(self):
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()
        super().__del__()

    def process(self, _) -> Item:
        socks, item = self._poll_prompt()
        ret, frame = self.cap.read()
        item.data["np_frame"] = frame
        return item


class FrameIngest(Stage):
    def __init__(self):
        self._inv_255 = 1.0 / 255.0

    @torch.inference_mode()
    def process(self, item: Item) -> Item:
        if "np_frame" in item.data:
            image = item.data["np_frame"]
        else:
            frame_msg: Frame = item.data["raw_frame"]
            image = decode_frame(frame_msg)
            image = np.array(image)

        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to("cuda", torch.bfloat16).mul(self._inv_255)
        item.data["frame_tensor"] = tensor
        return item


class Diffusion(Stage):
    def __init__(self, pipe: StableDiffusionImg2ImgPipeline):
        self.pipe = pipe
        self.rng = torch.Generator(device="cuda").manual_seed(42)

    def process(self, item: Item) -> Item:
        image = item.data["frame_tensor"]
        prompt_kwargs = item.data["prompt_kwargs"]
        latents = self.pipe.__call__(
            image=image,
            num_inference_steps=2,
            guidance_scale=0.5,
            strength=0.73,
            generator=self.rng.clone_state(),
            output_type="latent",
            **prompt_kwargs,
        ).images
        item.data["latents"] = latents
        return item


class DecoderAndUpscale(Stage):
    def __init__(self, decoder: FusedDecodeUpscale):
        self.decoder = decoder

    def process(self, item: Item) -> Item:
        latents = item.data["latents"]
        decoded = self.decoder(latents)
        item.data["decoded_image"] = decoded
        return item


class SmoothLatents(Stage):
    def __init__(self, alpha: float = 0.66):
        self.alpha = alpha
        self.prev_latents: torch.Tensor | None = None

    def process(self, item: Item) -> Item:
        latents = item.data["latents"]
        if self.prev_latents is None:
            self.prev_latents = latents
        else:
            latents = self.alpha * latents + (1 - self.alpha) * self.prev_latents
            self.prev_latents = latents
        item.data["latents"] = latents
        return item


class DeviceHostTransfer(Stage):
    @torch.inference_mode()
    def process(self, item: Item) -> Item:
        cuda_image = item.data["decoded_image"]
        final_image = torch_to_np(cuda_image)
        item.data["final_image"] = final_image
        return item


class IpcFramePublisher(Stage):
    def __init__(self, config: Config, exit_stack: ExitStack):
        os.makedirs(config.ipc_dir, exist_ok=True)
        self.pub_socket = create_socket(
            zmq.PUB, config.ipc_ui_diffusion_addr, "bind", exit_stack=exit_stack
        )

    def process(self, item: Item) -> Item:
        image = item.data["final_image"]
        msg = Frame(data=encode_frame(image))
        self.pub_socket.send(bytes(msg))
        return item


@app.command()
def main(
    device_id: str = "0",
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    fmt: str = "MJPG",
    convert_rgb: bool = True,
    matmul_precision: str = "medium",
    camera_mode: Literal["native", "zmq", "zmq+process"] = "zmq",
    log_level: str = "INFO",
):
    torch.set_float32_matmul_precision(matmul_precision)
    cap_config = {
        "device_id": device_id,
        "width": width,
        "height": height,
        "fps": fps,
        "fmt": fmt,
        "convert_rgb": convert_rgb,
        "log_level": "WARNING",
    }
    config = Config()
    logging.basicConfig(level=log_level, format="%(levelname)s\t%(name)s\t%(message)s")
    pipe, decoder = get_pipeline("stabilityai/sd-turbo", upscale=None)

    os.makedirs(config.ipc_dir, exist_ok=True)

    with ExitStack() as exit_stack:
        prompt_pub = create_socket(
            zmq.PUB, config.ipc_ui_prompt_addr, "bind", exit_stack=exit_stack
        )

        frame_capture = init_frame_source(
            cap_config=cap_config,
            config=config,
            camera_mode=camera_mode,
            pipe=pipe,
            exit_stack=exit_stack,
            prompt_republish=prompt_pub,
        )

        frame_publisher = IpcFramePublisher(config, exit_stack)

        pipeline = (
            count()
            | pl.thread.map(frame_capture, workers=1, maxsize=1)
            | pl.thread.map(FrameIngest(), workers=1, maxsize=1)
            | pl.thread.map(Diffusion(pipe), workers=1, maxsize=1)
            | pl.thread.map(SmoothLatents(), workers=1, maxsize=1)
            | pl.thread.map(DecoderAndUpscale(decoder), workers=1, maxsize=2)
            | pl.thread.map(DeviceHostTransfer(), workers=1)
            | pl.thread.map(frame_publisher, workers=1)
        )

        run_pipeline(pipeline)


def run_pipeline(pipeline: Iterable[Item]) -> NoReturn:
    for _ in tqdm(pipeline):
        pass


def init_frame_source(
    cap_config: dict[str, str | int | bool],
    config: Config,
    camera_mode: Literal["native", "zmq", "zmq+process"],
    pipe: StableDiffusionImg2ImgPipeline,
    exit_stack: ExitStack,
    prompt_republish: zmq.Socket | None = None,
) -> Stage:
    if camera_mode == "native":
        return NativeFrameCapture(
            config=config,
            pipe=pipe,
            cap_config=cap_config,
            prompt_republish=prompt_republish,
        )
    frame_capture = ZmqFrameCapture(config=config, pipe=pipe, prompt_republish=prompt_republish)
    if camera_mode == "zmq+process":
        p = multiprocessing.Process(
            target=mirror.native_camera.main,
            kwargs=cap_config,
            name="camera-capture",
        )
        exit_stack.callback(p.kill)
        atexit.register(p.kill)
        p.start()
    return frame_capture


if __name__ == "__main__":
    app()
