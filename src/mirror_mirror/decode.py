import io
import logging

import PIL.Image
import cv2
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


def decode_frame(frame: bytes) -> npt.NDArray[np.uint8]:
    """Decode JPEG bytes to numpy array"""
    try:
        # Convert bytes to PIL Image
        image = PIL.Image.open(io.BytesIO(frame)).convert("RGB")
        # Convert to numpy array
        return np.array(image, dtype=np.uint8)
    except Exception as e:
        logger.error(f"Failed to decode frame: {e}")
        raise


def encode_frame(frame: npt.NDArray[np.uint8]) -> bytes:
    """Encode OpenCV frame to JPEG bytes"""
    try:
        # Ensure the frame is in the right format
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame
            
        success, data = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not success:
            logger.error("Failed to encode frame")
            raise RuntimeError("Failed to encode frame")
        return data.tobytes()
    except Exception as e:
        logger.error(f"Failed to encode frame: {e}")
        raise
