import logging
from contextlib import ExitStack
from itertools import count
from pprint import pformat
from typing import Any, LiteralString, NoReturn

import cv2
import typer
import zmq
from tqdm import tqdm
from typer import Typer

from mirror.config import config
from mirror.generated.messages import Frame
from mirror.logger import logger
from mirror.zmq_utils import create_socket

app = Typer()


@app.command()
def main(
    device_id: str = "0",
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    convert_rgb: bool = True,
    fmt: str = "MJPG",
    backend: str | None = None,
    log_level: str = "INFO",
) -> NoReturn:
    logging.basicConfig(level=log_level)
    logger.info("Native camera module started.")
    with ExitStack() as exit_stack:
        frame_publisher = create_socket(
            zmq.PUB, config.zmq_frames_bind_addr, "bind", conflate=True, exit_stack=exit_stack
        )
        capture_config = {
            "device_id": device_id,
            "width": width,
            "height": height,
            "fps": fps,
            "convert_rgb": convert_rgb,
            "fmt": fmt,
            "backend": backend,
            "log_level": log_level,
        }
        cap = init_camera(capture_config)
        exit_stack.callback(cap.release)

        loop(cap, frame_publisher, capture_config)


def loop(cap: cv2.VideoCapture, frame_pub: zmq.Socket, cap_config: dict) -> NoReturn:
    packet = Frame()
    disable_progressbar = (
        logging.getLevelNamesMapping()[cap_config.get("log_level", "INFO")] == logging.WARNING
    )
    for i in tqdm(count(), unit="frame", disable=disable_progressbar):
        ret, frame = cap.read()
        if not ret:
            logger.error("Camera frame read failed. resetting!")
            cap = init_camera(cap_config)
        frame = cv2.resize(frame, (config.frame_size, config.frame_size))
        frame = cv2.flip(frame, 0)
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        frame_bytes = buffer.tobytes()
        packet.data = frame_bytes
        frame_pub.send(bytes(packet))


def init_camera(cap_config: dict[str, Any]) -> cv2.VideoCapture:
    backend = cap_config.get("backend")
    backends: dict[LiteralString, int] = {
        "ffmpeg": cv2.CAP_FFMPEG,
        "v4l2": cv2.CAP_V4L2,
    }
    backend = backends.get(backend)

    device_id = str(cap_config["device_id"])
    if device_id.isnumeric():
        device_id = int(device_id)
    width = cap_config["width"]
    height = cap_config["height"]
    fps = cap_config["fps"]
    fmt = cap_config["fmt"]
    convert_rgb = cap_config["convert_rgb"]

    if backend:
        cap = cv2.VideoCapture(device_id, backend)
    else:
        cap = cv2.VideoCapture(device_id)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fmt))
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 1 if convert_rgb else 0)
    if not cap.isOpened():
        logger.error("Failed to open camera device %s", device_id)
        raise typer.Exit(code=1)
    logger.info("Camera device %d opened successfully", device_id)
    backend_name = cap.getBackendName()
    actual_config = {
        "device_id": device_id,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": int(cap.get(cv2.CAP_PROP_FPS)),
        "backend": backend_name,
        "fmt": int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, byteorder="little").decode(),
    }

    logger.info("Camera config: %s", pformat(actual_config))
    return cap


if __name__ == "__main__":
    app()
