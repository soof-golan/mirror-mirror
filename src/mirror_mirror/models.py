from typing import Literal
import numpy as np
import numpy.typing as npt

from pydantic import BaseModel, Field


class FrameMessage(BaseModel):
    tag: Literal["frame"] = "frame"
    frame: bytes  # JPEG encoded frame
    timestamp: float
    camera_id: int = 0


class PromptMessage(BaseModel):
    tag: Literal["prompt"] = "prompt"
    prompt: str
    timestamp: float


class AudioMessage(BaseModel):
    tag: Literal["audio"] = "audio"
    audio_data: bytes  # Raw audio bytes
    sample_rate: int
    timestamp: float


class LatentsMessage(BaseModel):
    tag: Literal["latents"] = "latents"
    latents: bytes  # Serialized numpy array
    shape: tuple[int, ...]
    dtype: str
    timestamp: float
    source: str  # "camera" or "diffusion"


class EmbeddingMessage(BaseModel):
    tag: Literal["embedding"] = "embedding"
    embedding: bytes  # Serialized numpy array
    shape: tuple[int, ...]
    dtype: str
    text: str
    timestamp: float


class ProcessedFrameMessage(BaseModel):
    tag: Literal["processed_frame"] = "processed_frame"
    frame: bytes  # JPEG encoded processed frame
    timestamp: float
    processing_time: float


class CarrierMessage(BaseModel):
    content: (
        FrameMessage 
        | PromptMessage 
        | AudioMessage 
        | LatentsMessage 
        | EmbeddingMessage 
        | ProcessedFrameMessage
    ) = Field(discriminator="tag")


def serialize_array(arr: npt.NDArray) -> tuple[bytes, tuple[int, ...], str]:
    """Serialize a numpy array to bytes with metadata"""
    return arr.tobytes(), arr.shape, str(arr.dtype)


def deserialize_array(data: bytes, shape: tuple[int, ...], dtype: str) -> npt.NDArray:
    """Deserialize bytes back to numpy array"""
    return np.frombuffer(data, dtype=dtype).reshape(shape)
