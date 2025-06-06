import pytest
import asyncio
import time
import logging
from unittest.mock import Mock, MagicMock, AsyncMock, patch
import numpy as np

from mirror_mirror.models import (
    FrameMessage, 
    LatentsMessage, 
    ProcessedFrameMessage,
    CarrierMessage,
    encode_bytes,
    decode_bytes,
    serialize_array,
    deserialize_array
)
from mirror_mirror.decode import encode_frame, decode_frame

logger = logging.getLogger(__name__)


@pytest.mark.integration
class TestPipelineDataFlow:
    """Test data flow through the pipeline components"""
    
    def test_frame_to_latents_flow(self, sample_frame):
        """Test the complete flow from frame to latents"""
        
        # Step 1: Create a frame like camera_server would
        encoded_frame = encode_frame(sample_frame)
        frame_message = FrameMessage(
            frame=encode_bytes(encoded_frame),
            timestamp=time.time(),
            camera_id=0
        )
        
        # Step 2: Verify we can decode the frame (like latent_encoder would)
        frame_bytes = decode_bytes(frame_message.frame)
        decoded_frame = decode_frame(frame_bytes)
        
        assert decoded_frame.shape == sample_frame.shape
        assert decoded_frame.dtype == sample_frame.dtype
        
        # Step 3: Simulate latent encoding
        mock_latents = np.random.randn(1, 4, 64, 64).astype(np.float32)
        latents_data, shape, dtype = serialize_array(mock_latents)
        
        latents_message = LatentsMessage(
            latents=latents_data,
            shape=shape,
            dtype=dtype,
            timestamp=frame_message.timestamp,
            source="camera"
        )
        
        # Step 4: Verify latents can be deserialized (like diffusion_server would)
        reconstructed_latents = deserialize_array(
            latents_message.latents,
            latents_message.shape,
            latents_message.dtype
        )
        
        np.testing.assert_array_equal(reconstructed_latents, mock_latents)
        
        logger.info(f"Successfully tested frame->latents flow: {sample_frame.shape} -> {shape}")
    
    def test_latents_to_processed_frame_flow(self, sample_frame):
        """Test flow from latents back to processed frame"""
        
        # Step 1: Start with latents (like diffusion_server output)
        mock_latents = np.random.randn(1, 4, 64, 64).astype(np.float32)
        latents_data, shape, dtype = serialize_array(mock_latents)
        
        diffused_latents_message = LatentsMessage(
            latents=latents_data,
            shape=shape,
            dtype=dtype,
            timestamp=time.time(),
            source="diffusion"
        )
        
        # Step 2: Simulate latent decoding (like latent_decoder would)
        latents_array = deserialize_array(
            diffused_latents_message.latents,
            diffused_latents_message.shape,
            diffused_latents_message.dtype
        )
        
        # Simulate VAE decoding to image
        mock_decoded_image = sample_frame  # Use sample frame as mock output
        encoded_processed_frame = encode_frame(mock_decoded_image)
        
        # Step 3: Create processed frame message
        processed_frame_message = ProcessedFrameMessage(
            frame=encode_bytes(encoded_processed_frame),
            timestamp=diffused_latents_message.timestamp,
            processing_time=0.123
        )
        
        # Step 4: Verify display component can decode the frame
        frame_bytes = decode_bytes(processed_frame_message.frame)
        final_frame = decode_frame(frame_bytes)
        
        assert final_frame.shape == sample_frame.shape
        assert final_frame.dtype == sample_frame.dtype
        
        logger.info(f"Successfully tested latents->frame flow: {shape} -> {final_frame.shape}")
    
    def test_complete_pipeline_simulation(self, sample_frame):
        """Test complete pipeline data flow simulation"""
        
        start_time = time.time()
        
        # Camera server simulation
        encoded_frame = encode_frame(sample_frame)
        frame_msg = FrameMessage(
            frame=encode_bytes(encoded_frame),
            timestamp=start_time,
            camera_id=0
        )
        
        logger.info(f"1. Camera: Generated frame message {len(frame_msg.frame)} chars")
        
        # Latent encoder simulation
        frame_bytes = decode_bytes(frame_msg.frame)
        decoded_frame = decode_frame(frame_bytes)
        
        # Mock latent encoding
        mock_latents = np.random.randn(1, 4, 64, 64).astype(np.float32)
        latents_data, shape, dtype = serialize_array(mock_latents)
        
        camera_latents_msg = LatentsMessage(
            latents=latents_data,
            shape=shape,
            dtype=dtype,
            timestamp=frame_msg.timestamp,
            source="camera"
        )
        
        logger.info(f"2. Latent Encoder: {decoded_frame.shape} -> {shape}")
        
        # Diffusion server simulation
        input_latents = deserialize_array(
            camera_latents_msg.latents,
            camera_latents_msg.shape,
            camera_latents_msg.dtype
        )
        
        # Mock diffusion (just add some noise)
        diffused_latents = input_latents + np.random.randn(*input_latents.shape) * 0.1
        diffused_data, _, _ = serialize_array(diffused_latents)
        
        diffused_latents_msg = LatentsMessage(
            latents=diffused_data,
            shape=shape,
            dtype=dtype,
            timestamp=camera_latents_msg.timestamp,
            source="diffusion"
        )
        
        logger.info(f"3. Diffusion: Applied diffusion to latents {shape}")
        
        # Latent decoder simulation
        output_latents = deserialize_array(
            diffused_latents_msg.latents,
            diffused_latents_msg.shape,
            diffused_latents_msg.dtype
        )
        
        # Mock VAE decoding (use original frame)
        mock_decoded_image = sample_frame
        encoded_final_frame = encode_frame(mock_decoded_image)
        
        processing_time = time.time() - start_time
        processed_msg = ProcessedFrameMessage(
            frame=encode_bytes(encoded_final_frame),
            timestamp=diffused_latents_msg.timestamp,
            processing_time=processing_time
        )
        
        logger.info(f"4. Latent Decoder: {shape} -> final frame")
        
        # Display simulation
        final_frame_bytes = decode_bytes(processed_msg.frame)
        final_frame = decode_frame(final_frame_bytes)
        
        logger.info(f"5. Display: Ready to show {final_frame.shape} frame")
        
        # Verify complete pipeline integrity
        assert final_frame.shape == sample_frame.shape
        assert final_frame.dtype == sample_frame.dtype
        assert processed_msg.processing_time > 0
        assert processed_msg.timestamp == start_time
        
        logger.info(f"Complete pipeline simulation successful in {processing_time:.3f}s")


@pytest.mark.integration
class TestMessageCarrierIntegration:
    """Test CarrierMessage with different message types"""
    
    def test_carrier_message_round_trip(self, frame_message, latents_message, prompt_message):
        """Test CarrierMessage serialization/deserialization round trip"""
        
        test_messages = [frame_message, latents_message, prompt_message]
        
        for original_msg in test_messages:
            # Create carrier
            carrier = CarrierMessage(content=original_msg)
            
            # Serialize to JSON
            json_data = carrier.model_dump()
            
            # Deserialize from JSON
            reconstructed_carrier = CarrierMessage.model_validate(json_data)
            
            # Verify content matches
            assert type(reconstructed_carrier.content) == type(original_msg)
            assert reconstructed_carrier.content.tag == original_msg.tag
            assert reconstructed_carrier.content.timestamp == original_msg.timestamp
            
            logger.info(f"Round trip test passed for {original_msg.tag}")
    
    def test_carrier_message_type_discrimination(self):
        """Test that CarrierMessage correctly discriminates message types"""
        
        # Create different message types
        frame_msg = FrameMessage(
            frame="test_frame_data",
            timestamp=time.time(),
            camera_id=0
        )
        
        latents_msg = LatentsMessage(
            latents="test_latents_data",
            shape=(1, 4, 64, 64),
            dtype="float32",
            timestamp=time.time(),
            source="camera"
        )
        
        # Wrap in carriers
        frame_carrier = CarrierMessage(content=frame_msg)
        latents_carrier = CarrierMessage(content=latents_msg)
        
        # Verify type discrimination works
        assert isinstance(frame_carrier.content, FrameMessage)
        assert isinstance(latents_carrier.content, LatentsMessage)
        
        assert frame_carrier.content.tag == "frame"
        assert latents_carrier.content.tag == "latents"
        
        logger.info("Message type discrimination test passed")


@pytest.mark.integration
class TestPerformanceIntegration:
    """Test performance of integrated components"""
    
    def test_pipeline_performance_simulation(self, sample_frame, performance_monitor):
        """Test performance of simulated pipeline"""
        
        performance_monitor.start("full_pipeline_simulation")
        
        # Simulate processing 10 frames through the pipeline
        for i in range(10):
            # Camera encoding
            encoded_frame = encode_frame(sample_frame)
            frame_msg = FrameMessage(
                frame=encode_bytes(encoded_frame),
                timestamp=time.time(),
                camera_id=0
            )
            
            # Latent encoder simulation
            frame_bytes = decode_bytes(frame_msg.frame)
            decoded_frame = decode_frame(frame_bytes)
            
            # Mock latent processing
            mock_latents = np.random.randn(1, 4, 64, 64).astype(np.float32)
            latents_data, shape, dtype = serialize_array(mock_latents)
            
            # Mock diffusion
            diffused_latents = mock_latents + np.random.randn(*mock_latents.shape) * 0.1
            diffused_data, _, _ = serialize_array(diffused_latents)
            
            # Mock decoder
            encoded_final = encode_frame(sample_frame)
            processed_msg = ProcessedFrameMessage(
                frame=encode_bytes(encoded_final),
                timestamp=frame_msg.timestamp,
                processing_time=0.1
            )
            
            # Mock display
            final_bytes = decode_bytes(processed_msg.frame)
            final_frame = decode_frame(final_bytes)
        
        duration = performance_monitor.end("full_pipeline_simulation")
        
        # Should process 10 frames reasonably quickly
        frames_per_second = 10 / duration
        assert frames_per_second > 1.0, f"Pipeline too slow: {frames_per_second:.2f} FPS"
        
        logger.info(f"Pipeline simulation: {frames_per_second:.2f} FPS for 10 frames")
    
    def test_memory_usage_simulation(self, sample_frame):
        """Test memory usage doesn't grow excessively"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Process many frames to check for memory leaks
        for i in range(100):
            # Simulate pipeline processing
            encoded = encode_frame(sample_frame)
            frame_msg = FrameMessage(
                frame=encode_bytes(encoded),
                timestamp=time.time(),
                camera_id=0
            )
            
            # Decode and re-encode
            frame_bytes = decode_bytes(frame_msg.frame)
            decoded = decode_frame(frame_bytes)
            
            # Create latents message
            mock_latents = np.random.randn(1, 4, 64, 64).astype(np.float32)
            latents_data, shape, dtype = serialize_array(mock_latents)
            latents_msg = LatentsMessage(
                latents=latents_data,
                shape=shape,
                dtype=dtype,
                timestamp=time.time(),
                source="camera"
            )
            
            # Clean up references
            del frame_msg, decoded, latents_msg
        
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory
        memory_growth_mb = memory_growth / (1024 * 1024)
        
        # Memory growth should be reasonable (less than 50MB for 100 frames)
        assert memory_growth_mb < 50, f"Excessive memory growth: {memory_growth_mb:.2f} MB"
        
        logger.info(f"Memory growth: {memory_growth_mb:.2f} MB for 100 frames") 