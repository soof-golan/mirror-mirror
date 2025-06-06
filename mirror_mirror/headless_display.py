import asyncio
import logging
import time
import cv2
from pathlib import Path
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
    output_dir: str = "output_frames"
    max_frames: int = 100  # Maximum frames to save
    save_interval: int = 1  # Save every Nth frame
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
        self.frame_count = 0
    
    def update(self, processing_time: float, timestamp: float):
        """Update performance metrics"""
        self.frame_count += 1
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
        stats = {'frame_count': self.frame_count}
        
        if self.processing_times:
            stats['avg_processing_time'] = sum(self.processing_times) / len(self.processing_times)
            stats['max_processing_time'] = max(self.processing_times)
        
        if self.frame_intervals:
            stats['avg_fps'] = 1.0 / (sum(self.frame_intervals) / len(self.frame_intervals))
            stats['avg_interval'] = sum(self.frame_intervals) / len(self.frame_intervals)
        
        return stats


class HeadlessDisplay:
    def __init__(self):
        self.output_dir = Path(config.output_dir)
        self.frame_count = 0
        self.performance_monitor = PerformanceMonitor()
        self.running = True
        
    def initialize(self):
        """Initialize output directory"""
        self.output_dir.mkdir(exist_ok=True)
        logger.info(f"Headless display initialized - saving to: {self.output_dir}")
        logger.info(f"Will save max {config.max_frames} frames (every {config.save_interval})")
        
    def display_frame(self, frame_msg: ProcessedFrameMessage):
        """Save processed frame to disk"""
        if not self.running:
            return
            
        try:
            # Decode frame
            frame = decode_frame(frame_msg.frame)
            
            # Update performance metrics
            self.performance_monitor.update(frame_msg.processing_time, frame_msg.timestamp)
            stats = self.performance_monitor.get_stats()
            
            # Log performance every 10 frames
            if stats['frame_count'] % 10 == 0:
                logger.info(f"Frame {stats['frame_count']}: "
                          f"FPS: {stats.get('avg_fps', 0):.1f}, "
                          f"Processing: {stats.get('avg_processing_time', 0)*1000:.1f}ms")
            
            # Save frame if within limits and interval
            if (stats['frame_count'] <= config.max_frames and 
                stats['frame_count'] % config.save_interval == 0):
                
                # Convert RGB to BGR for OpenCV saving
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Save with timestamp and frame number
                timestamp_str = f"{frame_msg.timestamp:.3f}"
                filename = f"frame_{stats['frame_count']:05d}_{timestamp_str}.jpg"
                filepath = self.output_dir / filename
                
                cv2.imwrite(str(filepath), frame_bgr)
                
                logger.info(f"Saved frame: {filename} "
                          f"(processing: {frame_msg.processing_time:.3f}s)")
            
            # Stop after max frames
            if stats['frame_count'] >= config.max_frames:
                logger.info(f"Reached maximum frames ({config.max_frames}), stopping")
                self.running = False
            
        except Exception as e:
            logger.error(f"Error saving frame: {e}")
    
    def cleanup(self):
        """Cleanup and show summary"""
        stats = self.performance_monitor.get_stats()
        logger.info(f"Headless display finished:")
        logger.info(f"  Total frames: {stats['frame_count']}")
        if 'avg_fps' in stats:
            logger.info(f"  Average FPS: {stats['avg_fps']:.1f}")
        if 'avg_processing_time' in stats:
            logger.info(f"  Average processing time: {stats['avg_processing_time']*1000:.1f}ms")
        logger.info(f"  Output directory: {self.output_dir}")


# Global display instance
display = HeadlessDisplay()


@broker.subscriber(channel=PubSub(config.channel))
async def save_processed_frames(
    carrier: CarrierMessage,
    _message: RedisMessage,
):
    """Save processed frames to disk"""
    
    if not isinstance(carrier.content, ProcessedFrameMessage):
        logger.warning(f"Unexpected message type: {type(carrier.content)}")
        return
    
    frame_msg = carrier.content
    
    # Initialize on first frame
    if display.frame_count == 0:
        display.initialize()
    
    # Save the frame
    display.display_frame(frame_msg)
    
    # Check if should stop
    if not display.running:
        logger.info("Headless display finished, stopping")
        # Could trigger app shutdown here


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info(f"Starting headless display on channel: {config.channel}")
    logger.info(f"Saving frames to: {config.output_dir}")
    
    try:
        asyncio.run(app.run())
    finally:
        display.cleanup() 