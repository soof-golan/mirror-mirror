import concurrent
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from io import BytesIO
from typing import Literal

import numpy as np
import PIL.Image
from PIL import Image

from mirror.generated.messages import Frame
from mirror.logger import logger


def decode_frame(frame_msg: Frame) -> Image.Image:
    """Decode JPEG frame to PIL Image."""
    return Image.open(BytesIO(frame_msg.data))


def encode_frame(
    image: Image.Image | np.ndarray[tuple[int, int, Literal[3]], np.dtype[np.uint8]],
) -> bytes:
    """Encode PIL Image to JPEG bytes."""
    if isinstance(image, np.ndarray):
        image = PIL.Image.fromarray(image)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    return buffer.getvalue()


def log_timing[F](func: F, name: str | None = None) -> F:
    alpha = 0.1
    geometric_log_spacing = 1.5
    log_every = 2

    qualname = name or func.__qualname__
    template = f"average time: %.3fs for {qualname} "

    @wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal log_every
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        wrapper.dt = wrapper.dt * (1 - alpha) + (end - start) * alpha
        wrapper.call_count += 1
        if wrapper.call_count % log_every == 0:
            log_every = int(log_every * geometric_log_spacing)
            logger.info(template, wrapper.dt)
        return result

    wrapper.dt = 0.5
    wrapper.call_count = 0
    return wrapper


def quit_on_error(fut: concurrent.futures.Future) -> None:
    if exc := fut.exception():
        logger.error("Fatal error in worker thread", exc_info=exc)
        exit(1)


class BoundedThreadPoolExecutor(ThreadPoolExecutor):
    def __init__(self, max_workers: int, max_queue_size: int):
        super().__init__(max_workers=max_workers)
        self._work_queue = queue.Queue(maxsize=max_queue_size)
