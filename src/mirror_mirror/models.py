from typing import Literal

from pydantic import BaseModel, Base64Bytes, Field


class FrameMessage(BaseModel):
    msg_type: Literal["frame"] = "frame"
    frame: Base64Bytes


class PromptMessage(BaseModel):
    msg_type: Literal["prompt"] = "prompt"
    prompt: str


class DiffuserMessage(BaseModel):
    msg: PromptMessage | FrameMessage = Field(discriminator="msg_type")
