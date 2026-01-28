import logging
from contextlib import ExitStack
from itertools import count
from typing import Any, NoReturn

import sounddevice as sd
import zmq
from sounddevice import DeviceList
from tqdm import tqdm
from typer import Typer

from mirror.config import config
from mirror.generated.messages import Audio
from mirror.zmq_utils import create_socket

app = Typer()

logger = logging.getLogger(__name__)


@app.command()
def main(
    lookup: str | None = "",
    device_id: int | None = None,
    log_level: str = "INFO",
    chunk_size_ms: int = 5,
    limit: int | None = None,
    setup_logging: bool = True,
) -> NoReturn:
    if setup_logging:
        logging.basicConfig(level=log_level)
    logger.setLevel(log_level)
    logger.info("Native microphone module started.")
    logger.info("Will publish audio chunks to %s", config.zmq_audio_bind_addr)
    logger.info("%s", sd.query_devices())
    device_list: DeviceList = sd.query_devices()
    if device_id is None:
        for i, device in enumerate(device_list):
            device: dict[str, Any]
            if lookup in device["name"].lower():
                device_id = i
                break

    if device_id is None:
        logger.exception("Could not find a device with name '%s'", lookup)

    with ExitStack() as exit_stack:
        audio_pub = create_socket(
            zmq.PUB, config.zmq_audio_bind_addr, "bind", conflate=False, exit_stack=exit_stack
        )
        stream = init_mic(device_id)
        exit_stack.enter_context(stream)
        loop(audio_pub, stream, chunk_size_ms=chunk_size_ms, limit=limit, device_id=device_id)


def loop(
    audio_pub: zmq.Socket,
    stream: sd.InputStream,
    chunk_size_ms: int,
    limit: int | None,
    device_id: int,
) -> NoReturn:
    n_frames = int(stream.samplerate * chunk_size_ms / 1000)
    sr = int(stream.samplerate)
    channels = int(stream.channels)
    packet = Audio(sample_rate=sr, channels=channels)
    gen = count() if limit is None else range(limit)
    for _ in tqdm(gen, unit="audio frame", disable=logger.level >= logging.WARNING):
        try:
            buffer, _ = stream.read(n_frames)
        except sd.PortAudioError:
            stream = init_mic(device_id=device_id)
        frame_bytes = buffer.tobytes()
        packet.data = frame_bytes
        audio_pub.send(bytes(packet))


def init_mic(device_id: int) -> sd.InputStream:
    stream = sd.InputStream(
        device=device_id,
    )
    device = sd.query_devices(device_id)
    logger.info("Using input device: %s", device["name"])
    actual_config = {
        "device_id": stream.device,
        "channels": stream.channels,
        "samplerate": stream.samplerate,
    }
    logger.info(f"Initialized microphone with config: {actual_config}")
    return stream


if __name__ == "__main__":
    app()
