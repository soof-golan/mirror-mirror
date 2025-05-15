import logging
from contextlib import AsyncExitStack
import cv2
from faststream.redis import RedisBroker
from pydantic_settings import BaseSettings
from tenacity import retry, retry_if_exception_type, wait_random

from mirror_mirror.common import log_errors
from mirror_mirror.decode import encode_frame
from mirror_mirror.models import FrameMessage, CarrierMessage

logger = logging.getLogger(__name__)


class Config(BaseSettings):
    redis_url: str = "redis://localhost:6379"
    camera_id: int = 0


config = Config()


@retry(retry=retry_if_exception_type(), wait=wait_random(max=5))
@log_errors
async def main():
    async with AsyncExitStack() as stack:
        broker = RedisBroker(url=config.redis_url)
        await stack.enter_async_context(broker)
        cap = cv2.VideoCapture(config.camera_id)
        stack.callback(cap.release)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera {config.camera_id}")

        while True:
            success, frame = cap.read()
            if not success:
                raise RuntimeError("Failed to capture frame")

            encoded_frame = encode_frame(frame)
            message = FrameMessage(frame=encoded_frame)

            await broker.publish(
                message=CarrierMessage(content=message),
                channel=f"frames:camera:{config.camera_id}",
            )


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
