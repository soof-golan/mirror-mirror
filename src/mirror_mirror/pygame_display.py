import asyncio
import logging
import pygame
import numpy as np
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
    window_name: str = "Mirror Mirror - Pygame"
    window_width: int = 640
    window_height: int = 480
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


class PygameDisplay:
    def __init__(self):
        self.screen = None
        self.clock = None
        self.font = None
        self.running = False
        self.performance_monitor = PerformanceMonitor()
        
    def initialize(self):
        """Initialize pygame display"""
        if self.screen is None:
            try:
                pygame.init()
                
                # Try to set SDL video driver for better compatibility
                import os
                if 'SDL_VIDEODRIVER' not in os.environ:
                    # Try X11 first, fallback to others
                    for driver in ['x11', 'fbcon', 'directfb', 'dummy']:
                        os.environ['SDL_VIDEODRIVER'] = driver
                        try:
                            pygame.display.init()
                            break
                        except:
                            continue
                
                self.screen = pygame.display.set_mode((config.window_width, config.window_height))
                pygame.display.set_caption(config.window_name)
                self.clock = pygame.time.Clock()
                
                # Initialize font for text overlay
                try:
                    self.font = pygame.font.Font(None, 36)
                except:
                    self.font = pygame.font.SysFont('monospace', 24)
                
                self.running = True
                logger.info(f"Pygame display initialized: {config.window_width}x{config.window_height}")
                
            except Exception as e:
                logger.error(f"Failed to initialize pygame display: {e}")
                # Fall back to headless mode
                self.screen = None
                raise
    
    def display_frame(self, frame_msg: ProcessedFrameMessage):
        """Display a processed frame"""
        if not self.running:
            return
            
        try:
            # Decode frame
            frame = decode_frame(frame_msg.frame)
            
            # Update performance metrics
            self.performance_monitor.update(frame_msg.processing_time, frame_msg.timestamp)
            
            if self.screen is not None:
                # Convert numpy array to pygame surface
                # OpenCV uses BGR, pygame uses RGB, our frame should be RGB
                frame_rgb = frame
                
                # Resize frame to fit window if needed
                if frame.shape[:2] != (config.window_height, config.window_width):
                    import cv2
                    frame_rgb = cv2.resize(frame_rgb, (config.window_width, config.window_height))
                
                # Convert to pygame surface
                # pygame expects (width, height, 3) but numpy is (height, width, 3)
                frame_transposed = np.transpose(frame_rgb, (1, 0, 2))
                surface = pygame.surfarray.make_surface(frame_transposed)
                
                # Clear screen and draw frame
                self.screen.fill((0, 0, 0))
                self.screen.blit(surface, (0, 0))
                
                # Add performance overlay if enabled
                if config.show_fps and self.font:
                    self._draw_performance_overlay()
                
                # Update display
                pygame.display.flip()
                self.clock.tick(60)  # Limit to 60 FPS
                
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    logger.info("Window close requested")
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        logger.info("Q key pressed, closing display")
                        self.running = False
                    elif event.key == pygame.K_f:
                        config.show_fps = not config.show_fps
                        logger.info(f"FPS display: {'ON' if config.show_fps else 'OFF'}")
            
            logger.debug(f"Displayed frame with {frame_msg.processing_time:.3f}s processing time")
            
        except Exception as e:
            logger.error(f"Error displaying frame: {e}")
    
    def _draw_performance_overlay(self):
        """Draw performance statistics on screen"""
        stats = self.performance_monitor.get_stats()
        
        overlay_texts = []
        if 'avg_fps' in stats:
            overlay_texts.append(f"FPS: {stats['avg_fps']:.1f}")
        if 'avg_processing_time' in stats:
            overlay_texts.append(f"Proc: {stats['avg_processing_time']*1000:.1f}ms")
        
        y_offset = 10
        for text in overlay_texts:
            try:
                text_surface = self.font.render(text, True, (0, 255, 0))
                self.screen.blit(text_surface, (10, y_offset))
                y_offset += 30
            except:
                pass  # Skip if font rendering fails
    
    def cleanup(self):
        """Cleanup pygame resources"""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.running = False


# Global display instance
display = PygameDisplay()


@broker.subscriber(channel=PubSub(config.channel))
async def display_processed_frames(
    carrier: CarrierMessage,
    _message: RedisMessage,
):
    """Display processed frames using pygame"""
    
    if not isinstance(carrier.content, ProcessedFrameMessage):
        logger.warning(f"Unexpected message type: {type(carrier.content)}")
        return
    
    frame_msg = carrier.content
    
    # Initialize display on first frame
    if display.screen is None:
        try:
            display.initialize()
        except Exception as e:
            logger.error(f"Failed to initialize display: {e}")
            return
    
    # Display the frame
    display.display_frame(frame_msg)
    
    # Check if display should close
    if not display.running:
        logger.info("Display requested shutdown")
        # Could trigger app shutdown here if needed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info(f"Starting pygame display on channel: {config.channel}")
    logger.info("Press 'q' to quit, 'f' to toggle FPS display")
    
    try:
        asyncio.run(app.run())
    finally:
        display.cleanup() 