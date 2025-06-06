import logging
import time
from contextlib import AsyncExitStack
import cv2
import asyncio
from typing import AsyncGenerator

from faststream.redis import RedisBroker
from pydantic_settings import BaseSettings
from tenacity import retry, retry_if_exception_type, wait_exponential, stop_after_attempt

from mirror_mirror.common import log_errors
from mirror_mirror.decode import encode_frame
from mirror_mirror.models import FrameMessage, CarrierMessage, encode_bytes

logger = logging.getLogger(__name__)


class Config(BaseSettings):
    redis_url: str = "redis://localhost:6379"
    camera_id: int = 0
    fps: int = 24
    frame_width: int = 640
    frame_height: int = 480


config = Config()


async def run_camera_loop(cap: cv2.VideoCapture) -> AsyncGenerator[CarrierMessage, None]:
    """
    Main camera loop as an async generator.
    Captures frames and yields them as CarrierMessages.
    """
    frame_interval = 1.0 / config.fps
    last_frame_time = 0
    frames_published = 0
    start_time = time.time()

    while True:
        current_time = time.time()
        
        # Rate limiting
        if current_time - last_frame_time < frame_interval:
            await asyncio.sleep(0.001)  # non-blocking sleep
            continue
            
        success, frame = cap.read()
        if not success:
            logger.warning("Failed to capture frame, retrying...")
            continue

        try:
            encoded_frame = encode_frame(frame)
            message = FrameMessage(
                frame=encode_bytes(encoded_frame),
                timestamp=current_time,
                camera_id=config.camera_id
            )

            yield CarrierMessage(content=message)
            
            last_frame_time = current_time
            frames_published += 1
            
            # Log status every 100 frames
            if frames_published > 0 and frames_published % 100 == 0:
                elapsed = current_time - start_time
                fps = frames_published / elapsed if elapsed > 0 else 0
                logger.info(f"Published {frames_published} frames, actual FPS: {fps:.2f}")
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}", exc_info=True)
            continue


@retry(
    retry=retry_if_exception_type(Exception), 
    wait=wait_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(5)
)
@log_errors
async def main():
    async with AsyncExitStack() as stack:
        broker = RedisBroker(url=config.redis_url)
        await stack.enter_async_context(broker)
        
        cap = cv2.VideoCapture(config.camera_id)
        stack.callback(cap.release)
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera {config.camera_id}")
        
        # Configure camera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)
        cap.set(cv2.CAP_PROP_FPS, config.fps)
        
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(
            f"Camera {config.camera_id} opened: {actual_width}x{actual_height} @ {actual_fps}fps"
        )

        async for message in run_camera_loop(cap):
            await broker.publish(
                message=message,
                channel=f"frames:camera:{config.camera_id}",
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
