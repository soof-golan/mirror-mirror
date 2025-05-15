import io
import logging

import PIL.Image
import cv2
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


def decode_frame(frame: bytes) -> npt.NDArray[np.uint8]:
    print("frame start", frame[:20])
    return np.array(PIL.Image.open(io.BytesIO(frame)).convert("RGB"))


def encode_frame(frame: npt.NDArray[np.uint8]) -> bytes:
    """Encode OpenCV frame to JPEG bytes"""
    success, data = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not success:
        logger.error("Failed to encode frame")
        raise RuntimeError("Failed to encode frame")
    value = data.tobytes()
    return value
