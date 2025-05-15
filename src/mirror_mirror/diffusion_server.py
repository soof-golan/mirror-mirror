from abc import ABC, abstractmethod
from functools import cache
from typing import Literal

import cv2
import numpy as np
import numpy.typing as npt
import faststream
from faststream import FastStream, Depends
from faststream.redis import RedisBroker, PubSub
from pydantic_settings import BaseSettings

from mirror_mirror.common import log_errors, assert_unreachable
from mirror_mirror.decode import decode_frame
from mirror_mirror.models import CarrierMessage, FrameMessage, PromptMessage


class Config(BaseSettings):
    redis_url: str = "redis://localhost:6379"
    mode: Literal["fake"] = "fake"


config = Config()

broker = RedisBroker(url=config.redis_url)
app = FastStream(broker=broker)


class Diffuser(ABC):
    prompt: str

    @abstractmethod
    def diffuse(self, frame: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        pass

    @abstractmethod
    def update_prompt(self, prompt: str) -> str:
        pass


class FakeDiffuser(Diffuser):
    def __init__(self):
        self.prompt = "Hello, world!"

    def update_prompt(self, prompt: str):
        self.prompt = prompt

    def diffuse(self, frame: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        cv2.putText(
            frame,
            self.prompt,
            (10, 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )
        return frame


@cache
def get_diffuser() -> FakeDiffuser:
    match config.mode:
        case "fake":
            return FakeDiffuser()
        case _:
            return assert_unreachable()


@broker.subscriber(channel=PubSub("frames:camera:{feed}", pattern=True))
@broker.subscriber(channel=PubSub("prompts:*", pattern=True))
@broker.publisher(channel="diffused_images")
@log_errors
def to_diffuse(
    carrier: CarrierMessage,
    logger: faststream.Logger,
    feed: str = faststream.Path(),
    diffuser: Diffuser = Depends(get_diffuser),
):
    match carrier.content:
        case FrameMessage(frame=frame):
            logger.info("feed %s. frame nbytes %s", feed, len(frame))
            frame = decode_frame(frame)
            diffused = diffuser.diffuse(frame)
            return diffused
        case PromptMessage(prompt=prompt):
            diffuser.update_prompt(prompt)
            return None

    return None
