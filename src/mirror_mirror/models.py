from typing import Literal
import base64
import numpy as np
import numpy.typing as npt

from pydantic import BaseModel, Field


class FrameMessage(BaseModel):
    tag: Literal["frame"] = "frame"
    frame: str  # Base64 encoded JPEG frame
    timestamp: float
    camera_id: int = 0


class PromptMessage(BaseModel):
    tag: Literal["prompt"] = "prompt"
    prompt: str
    timestamp: float


class AudioMessage(BaseModel):
    tag: Literal["audio"] = "audio"
    audio_data: str  # Base64 encoded audio bytes
    sample_rate: int
    timestamp: float


class LatentsMessage(BaseModel):
    tag: Literal["latents"] = "latents"
    latents: str  # Base64 encoded serialized numpy array
    shape: tuple[int, ...]
    dtype: str
    timestamp: float
    source: str  # "camera" or "diffusion"


class EmbeddingMessage(BaseModel):
    tag: Literal["embedding"] = "embedding"
    embedding: str  # Base64 encoded serialized numpy array
    shape: tuple[int, ...]
    dtype: str
    text: str
    timestamp: float


class ProcessedFrameMessage(BaseModel):
    tag: Literal["processed_frame"] = "processed_frame"
    frame: str  # Base64 encoded JPEG processed frame
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


def encode_bytes(data: bytes) -> str:
    """Encode bytes to base64 string for JSON serialization"""
    return base64.b64encode(data).decode('utf-8')


def decode_bytes(data: str) -> bytes:
    """Decode base64 string back to bytes"""
    return base64.b64decode(data.encode('utf-8'))


def serialize_array(arr: npt.NDArray) -> tuple[str, tuple[int, ...], str]:
    """Serialize a numpy array to base64 string with metadata"""
    return encode_bytes(arr.tobytes()), arr.shape, str(arr.dtype)


def deserialize_array(data: str, shape: tuple[int, ...], dtype: str) -> npt.NDArray:
    """Deserialize base64 string back to numpy array"""
    bytes_data = decode_bytes(data)
    return np.frombuffer(bytes_data, dtype=dtype).reshape(shape)
