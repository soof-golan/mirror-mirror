import pytest
import asyncio
import time
import cv2
import numpy as np
from unittest.mock import Mock, MagicMock, AsyncMock, patch
import logging

from mirror_mirror.camera_server import Config, main
from mirror_mirror.models import FrameMessage, CarrierMessage, decode_bytes
from mirror_mirror.decode import decode_frame

logger = logging.getLogger(__name__)


@pytest.mark.unit
class TestCameraServerConfig:
    """Test camera server configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = Config()
        
        assert config.redis_url == "redis://localhost:6379"
        assert config.camera_id == 0
        assert config.fps == 24
        assert config.frame_width == 640
        assert config.frame_height == 480
        
        logger.info(f"Default config: {config}")
    
    def test_config_override(self):
        """Test configuration override from environment"""
        config = Config(
            redis_url="redis://test:6379",
            camera_id=1,
            fps=30,
            frame_width=1280,
            frame_height=720
        )
        
        assert config.redis_url == "redis://test:6379"
        assert config.camera_id == 1
        assert config.fps == 30
        assert config.frame_width == 1280
        assert config.frame_height == 720


@pytest.mark.unit
class TestCameraServerMocking:
    """Test camera server with mocked dependencies"""
    
    @pytest.fixture
    def mock_cv2_videocapture(self):
        """Mock cv2.VideoCapture"""
        with patch('mirror_mirror.camera_server.cv2.VideoCapture') as mock_vc:
            mock_cap = MagicMock()
            mock_vc.return_value = mock_cap
            
            # Configure mock camera
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: {
                cv2.CAP_PROP_FRAME_WIDTH: 640.0,
                cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
                cv2.CAP_PROP_FPS: 24.0
            }.get(prop, 0.0)
            
            # Mock successful frame reads
            test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            test_frame[:, :, 0] = 128  # Add some data
            mock_cap.read.return_value = (True, test_frame)
            
            yield mock_cap
    
    @pytest.fixture  
    def mock_redis_broker(self):
        """Mock Redis broker"""
        with patch('mirror_mirror.camera_server.RedisBroker') as mock_broker_class:
            mock_broker = AsyncMock()
            mock_broker_class.return_value = mock_broker
            
            # Mock context manager
            mock_broker.__aenter__ = AsyncMock(return_value=mock_broker)
            mock_broker.__aexit__ = AsyncMock(return_value=None)
            
            yield mock_broker
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing"""
        with patch('mirror_mirror.camera_server.config') as mock_cfg:
            mock_cfg.redis_url = "redis://test:6379"
            mock_cfg.camera_id = 0
            mock_cfg.fps = 5  # Lower FPS for faster testing
            mock_cfg.frame_width = 640
            mock_cfg.frame_height = 480
            yield mock_cfg

    async def test_camera_initialization_success(self, mock_cv2_videocapture, mock_redis_broker, mock_config):
        """Test successful camera initialization"""
        
        # Cancel after a short time to avoid infinite loop
        async def cancel_after_delay():
            await asyncio.sleep(0.1)
            raise KeyboardInterrupt()
        
        # Run both main and cancellation concurrently
        with pytest.raises(KeyboardInterrupt):
            await asyncio.gather(
                main(),
                cancel_after_delay(),
                return_exceptions=True
            )
        
        # Verify camera was opened
        mock_cv2_videocapture.isOpened.assert_called()
        
        # Verify camera configuration
        assert mock_cv2_videocapture.set.call_count >= 3  # Width, height, FPS
        
        # Verify Redis broker was used
        mock_redis_broker.publish.assert_called()
        
        logger.info("Camera initialization test completed successfully")

    async def test_camera_open_failure(self, mock_redis_broker, mock_config):
        """Test camera open failure"""
        
        with patch('mirror_mirror.camera_server.cv2.VideoCapture') as mock_vc:
            mock_cap = MagicMock()
            mock_vc.return_value = mock_cap
            mock_cap.isOpened.return_value = False  # Simulate failure
            
            with pytest.raises(RuntimeError, match="Failed to open camera"):
                await main()
    
    async def test_frame_read_failure(self, mock_cv2_videocapture, mock_redis_broker, mock_config):
        """Test handling of frame read failures"""
        
        # First few reads fail, then succeed
        mock_cv2_videocapture.read.side_effect = [
            (False, None),  # Fail
            (False, None),  # Fail
            (True, np.zeros((480, 640, 3), dtype=np.uint8))  # Success
        ]
        
        # Cancel after short time
        async def cancel_after_delay():
            await asyncio.sleep(0.1)
            raise KeyboardInterrupt()
        
        with pytest.raises(KeyboardInterrupt):
            await asyncio.gather(
                main(),
                cancel_after_delay(),
                return_exceptions=True
            )
        
        # Should have tried to read multiple times
        assert mock_cv2_videocapture.read.call_count >= 2

    async def test_frame_publishing(self, mock_cv2_videocapture, mock_redis_broker, mock_config):
        """Test frame publishing to Redis"""
        
        # Create a recognizable test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_frame[100:200, 100:200, :] = [255, 0, 0]  # Red square
        mock_cv2_videocapture.read.return_value = (True, test_frame)
        
        # Cancel after short time to collect some frames
        async def cancel_after_delay():
            await asyncio.sleep(0.2)
            raise KeyboardInterrupt()
        
        with pytest.raises(KeyboardInterrupt):
            await asyncio.gather(
                main(),
                cancel_after_delay(),
                return_exceptions=True
            )
        
        # Verify frames were published
        assert mock_redis_broker.publish.call_count > 0
        
        # Check the published message structure
        publish_calls = mock_redis_broker.publish.call_args_list
        assert len(publish_calls) > 0
        
        # Examine first published message
        first_call = publish_calls[0]
        args, kwargs = first_call
        
        assert 'message' in kwargs
        assert 'channel' in kwargs
        assert kwargs['channel'] == f"frames:camera:{mock_config.camera_id}"
        
        # Verify message structure
        carrier_msg = kwargs['message']
        assert isinstance(carrier_msg, CarrierMessage)
        assert isinstance(carrier_msg.content, FrameMessage)
        
        frame_msg = carrier_msg.content
        assert frame_msg.tag == "frame"
        assert frame_msg.camera_id == mock_config.camera_id
        assert isinstance(frame_msg.timestamp, float)
        assert isinstance(frame_msg.frame, str)  # Base64 encoded
        
        # Verify we can decode the frame back
        frame_bytes = decode_bytes(frame_msg.frame)
        decoded_frame = decode_frame(frame_bytes)
        assert decoded_frame.shape == test_frame.shape
        
        logger.info(f"Published {len(publish_calls)} frames successfully")

    async def test_fps_limiting(self, mock_cv2_videocapture, mock_redis_broker, mock_config):
        """Test FPS limiting functionality"""
        
        # Set very low FPS for testing
        mock_config.fps = 2
        
        start_time = time.time()
        
        # Cancel after enough time for multiple frames
        async def cancel_after_delay():
            await asyncio.sleep(1.0)  # 1 second should allow ~2 frames at 2 FPS
            raise KeyboardInterrupt()
        
        with pytest.raises(KeyboardInterrupt):
            await asyncio.gather(
                main(),
                cancel_after_delay(),
                return_exceptions=True
            )
        
        elapsed = time.time() - start_time
        publish_count = mock_redis_broker.publish.call_count
        
        # Should have published roughly 2 frames in 1 second
        assert publish_count <= 4, f"Too many frames published: {publish_count} in {elapsed:.2f}s"
        assert publish_count >= 1, f"Too few frames published: {publish_count} in {elapsed:.2f}s"
        
        logger.info(f"FPS limiting test: {publish_count} frames in {elapsed:.2f}s")


@pytest.mark.unit
class TestCameraServerPerformance:
    """Test camera server performance characteristics"""
    
    def test_frame_encoding_performance(self, sample_frame, performance_monitor):
        """Test frame encoding performance"""
        from mirror_mirror.decode import encode_frame
        
        performance_monitor.start("frame_encoding")
        
        # Encode multiple frames
        for _ in range(10):
            encoded = encode_frame(sample_frame)
            assert len(encoded) > 0
        
        duration = performance_monitor.end("frame_encoding")
        
        # Should be fast
        assert duration < 1.0, f"Frame encoding too slow: {duration}s for 10 frames"
        
        logger.info(f"Encoded 10 frames in {duration:.3f}s")
    
    def test_message_creation_performance(self, sample_frame_jpeg, performance_monitor):
        """Test message creation performance"""
        from mirror_mirror.models import encode_bytes
        
        performance_monitor.start("message_creation")
        
        # Create multiple messages
        for i in range(100):
            message = FrameMessage(
                frame=encode_bytes(sample_frame_jpeg),
                timestamp=time.time(),
                camera_id=0
            )
            carrier = CarrierMessage(content=message)
        
        duration = performance_monitor.end("message_creation") 
        
        # Should be very fast
        assert duration < 0.5, f"Message creation too slow: {duration}s for 100 messages"
        
        logger.info(f"Created 100 messages in {duration:.3f}s")


@pytest.mark.unit
class TestCameraServerErrorHandling:
    """Test error handling in camera server"""
    
    async def test_encoding_error_handling(self, mock_redis_broker, mock_config):
        """Test handling of frame encoding errors"""
        
        with patch('mirror_mirror.camera_server.cv2.VideoCapture') as mock_vc:
            mock_cap = MagicMock()
            mock_vc.return_value = mock_cap
            mock_cap.isOpened.return_value = True
            
            # Return invalid frame data that will cause encoding to fail
            mock_cap.read.return_value = (True, None)  # None frame should cause error
            
            # Mock encode_frame to raise an exception
            with patch('mirror_mirror.camera_server.encode_frame', side_effect=Exception("Encoding failed")):
                
                async def cancel_after_delay():
                    await asyncio.sleep(0.1)
                    raise KeyboardInterrupt()
                
                with pytest.raises(KeyboardInterrupt):
                    await asyncio.gather(
                        main(),
                        cancel_after_delay(),
                        return_exceptions=True
                    )
                
                # Should have attempted to read frames but not published due to errors
                assert mock_cap.read.call_count > 0
                # Should not have published anything due to encoding errors
                assert mock_redis_broker.publish.call_count == 0

    async def test_redis_publish_error_handling(self, mock_cv2_videocapture, mock_config):
        """Test handling of Redis publish errors"""
        
        with patch('mirror_mirror.camera_server.RedisBroker') as mock_broker_class:
            mock_broker = AsyncMock()
            mock_broker_class.return_value = mock_broker
            
            # Mock context manager
            mock_broker.__aenter__ = AsyncMock(return_value=mock_broker)
            mock_broker.__aexit__ = AsyncMock(return_value=None)
            
            # Make publish fail
            mock_broker.publish.side_effect = Exception("Redis connection failed")
            
            async def cancel_after_delay():
                await asyncio.sleep(0.1)
                raise KeyboardInterrupt()
            
            with pytest.raises(KeyboardInterrupt):
                await asyncio.gather(
                    main(),
                    cancel_after_delay(),
                    return_exceptions=True
                )
            
            # Should have attempted to publish
            assert mock_broker.publish.call_count > 0 