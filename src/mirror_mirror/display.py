import logging
import cv2
from faststream import FastStream
from faststream.redis import RedisBroker, PubSub
from faststream.redis.message import RedisMessage
from pydantic_settings import BaseSettings

from mirror_mirror.decode import decode_frame
from mirror_mirror.models import CarrierMessage, ProcessedFrameMessage

logger = logging.getLogger(__name__)


class Config(BaseSettings):
    redis_url: str = "redis://localhost:6379"
    channel: str = "images:processed"
    window_name: str = "Mirror Mirror"
    show_fps: bool = True


config = Config()
broker = RedisBroker(url=config.redis_url)
app = FastStream(broker=broker)


class PerformanceMonitor:
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.processing_times = []
        self.last_timestamp = None
        self.frame_intervals = []
    
    def update(self, processing_time: float, timestamp: float):
        """Update performance metrics"""
        self.processing_times.append(processing_time)
        if len(self.processing_times) > self.window_size:
            self.processing_times.pop(0)
        
        if self.last_timestamp is not None:
            interval = timestamp - self.last_timestamp
            self.frame_intervals.append(interval)
            if len(self.frame_intervals) > self.window_size:
                self.frame_intervals.pop(0)
        
        self.last_timestamp = timestamp
    
    def get_stats(self) -> dict:
        """Get current performance statistics"""
        stats = {}
        
        if self.processing_times:
            stats['avg_processing_time'] = sum(self.processing_times) / len(self.processing_times)
            stats['max_processing_time'] = max(self.processing_times)
        
        if self.frame_intervals:
            stats['avg_fps'] = 1.0 / (sum(self.frame_intervals) / len(self.frame_intervals))
            stats['avg_interval'] = sum(self.frame_intervals) / len(self.frame_intervals)
        
        return stats


performance_monitor = PerformanceMonitor()


@broker.subscriber(channel=PubSub(config.channel))
async def display_processed_frames(
    carrier: CarrierMessage,
    _message: RedisMessage,
):
    """Display processed frames with performance overlay"""
    
    if not isinstance(carrier.content, ProcessedFrameMessage):
        logger.warning(f"Unexpected message type: {type(carrier.content)}")
        return
    
    frame_msg = carrier.content
    
    try:
        # Decode frame
        frame = decode_frame(frame_msg.frame)
        
        # Update performance metrics
        performance_monitor.update(frame_msg.processing_time, frame_msg.timestamp)
        
        # Add performance overlay if enabled
        if config.show_fps:
            stats = performance_monitor.get_stats()
            
            # Prepare text overlay
            overlay_text = []
            if 'avg_fps' in stats:
                overlay_text.append(f"FPS: {stats['avg_fps']:.1f}")
            if 'avg_processing_time' in stats:
                overlay_text.append(f"Proc: {stats['avg_processing_time']*1000:.1f}ms")
            
            # Draw text overlay
            y_offset = 30
            for text in overlay_text:
                cv2.putText(
                    frame,
                    text,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),  # Green text
                    2,
                    cv2.LINE_AA
                )
                y_offset += 30
        
        # Display frame
        cv2.imshow(config.window_name, frame)
        
        # Handle window events
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            logger.info("Quit key pressed, closing display")
            cv2.destroyAllWindows()
            return
        elif key == ord('f'):
            # Toggle FPS display
            config.show_fps = not config.show_fps
            logger.info(f"FPS display: {'ON' if config.show_fps else 'OFF'}")
        
        logger.debug(f"Displayed frame with {frame_msg.processing_time:.3f}s processing time")
        
    except Exception as e:
        logger.error(f"Error displaying frame: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info(f"Starting display on channel: {config.channel}")
    logger.info("Press 'q' to quit, 'f' to toggle FPS display")
    
    try:
        asyncio.run(app.run())
    finally:
        cv2.destroyAllWindows()
