from typing import Literal

from pydantic import BaseModel, Base64Bytes, Field


class FrameMessage(BaseModel):
    tag: Literal["frame"] = "frame"
    frame: Base64Bytes


class PromptMessage(BaseModel):
    tag: Literal["prompt"] = "prompt"
    prompt: str


class CarrierMessage(BaseModel):
    content: PromptMessage | FrameMessage = Field(discriminator="tag")
