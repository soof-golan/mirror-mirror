import io

import PIL.Image
import numpy as np
import numpy.typing as npt


def decode_frame(frame: bytes) -> npt.NDArray[np.uint8]:
    return np.array(PIL.Image.open(io.BytesIO(frame)).convert("RGB"))


def encode_frame(frame: npt.NDArray[np.uint8]) -> bytes:
    """Encode OpenCV frame to JPEG bytes"""
    bio = io.BytesIO()
    PIL.Image.fromarray(frame).save(bio, format="JPEG")
    return bio.getvalue()
