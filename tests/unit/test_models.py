import pytest
import numpy as np
import time
import logging

from mirror_mirror import (
    FrameMessage,
    PromptMessage,
    LatentsMessage,
    ProcessedFrameMessage,
    CarrierMessage,
    encode_bytes,
    decode_bytes,
    serialize_array,
    deserialize_array,
)

logger = logging.getLogger(__name__)


@pytest.mark.unit
class TestMessageSerialization:
    """Test message serialization and deserialization"""

    def test_encode_decode_bytes(self):
        """Test bytes encoding and decoding"""
        original_data = b"Hello, World! \x00\x01\x02\x03"

        # Encode to base64 string
        encoded = encode_bytes(original_data)
        assert isinstance(encoded, str)
        assert len(encoded) > 0

        # Decode back to bytes
        decoded = decode_bytes(encoded)
        assert isinstance(decoded, bytes)
        assert decoded == original_data

        logger.info(f"Successfully encoded/decoded {len(original_data)} bytes")

    def test_serialize_deserialize_array(self):
        """Test numpy array serialization"""
        # Test different array types and shapes
        test_arrays = [
            np.array([1, 2, 3, 4], dtype=np.int32),
            np.random.randn(10, 10).astype(np.float32),
            np.zeros((2, 3, 4, 5), dtype=np.uint8),
            np.ones((1, 4, 64, 64), dtype=np.float16),
        ]

        for original_array in test_arrays:
            # Serialize
            data_str, shape, dtype_str = serialize_array(original_array)

            assert isinstance(data_str, str)
            assert shape == original_array.shape
            assert dtype_str == str(original_array.dtype)

            # Deserialize
            reconstructed = deserialize_array(data_str, shape, dtype_str)

            assert isinstance(reconstructed, np.ndarray)
            assert reconstructed.shape == original_array.shape
            assert reconstructed.dtype == original_array.dtype
            np.testing.assert_array_equal(reconstructed, original_array)

            logger.info(f"Successfully serialized array {shape} {dtype_str}")


@pytest.mark.unit
class TestFrameMessage:
    """Test FrameMessage functionality"""

    def test_frame_message_creation(self, sample_frame_jpeg):
        """Test creating a FrameMessage"""
        timestamp = time.time()
        camera_id = 1

        message = FrameMessage(frame=encode_bytes(sample_frame_jpeg), timestamp=timestamp, camera_id=camera_id)

        assert message.tag == "frame"
        assert message.timestamp == timestamp
        assert message.camera_id == camera_id
        assert isinstance(message.frame, str)

        # Verify we can decode the frame back
        decoded_frame = decode_bytes(message.frame)
        assert decoded_frame == sample_frame_jpeg

        logger.info(f"Created FrameMessage with {len(sample_frame_jpeg)} byte frame")

    def test_frame_message_defaults(self, sample_frame_jpeg):
        """Test FrameMessage with default values"""
        message = FrameMessage(frame=encode_bytes(sample_frame_jpeg), timestamp=time.time())

        assert message.camera_id == 0  # Default value
        assert message.tag == "frame"


@pytest.mark.unit
class TestLatentsMessage:
    """Test LatentsMessage functionality"""

    def test_latents_message_creation(self, sample_latents):
        """Test creating a LatentsMessage"""
        timestamp = time.time()
        source = "camera"

        data_str, shape, dtype_str = serialize_array(sample_latents)

        message = LatentsMessage(latents=data_str, shape=shape, dtype=dtype_str, timestamp=timestamp, source=source)

        assert message.tag == "latents"
        assert message.timestamp == timestamp
        assert message.source == source
        assert message.shape == sample_latents.shape
        assert message.dtype == str(sample_latents.dtype)

        # Verify we can deserialize the latents
        reconstructed = deserialize_array(message.latents, message.shape, message.dtype)
        np.testing.assert_array_equal(reconstructed, sample_latents)

        logger.info(f"Created LatentsMessage with shape {shape}")

    def test_latents_message_validation(self):
        """Test LatentsMessage validation"""
        # Currently LatentsMessage doesn't validate source values
        # This test documents the current behavior
        message = LatentsMessage(
            latents="test",
            shape=(1, 4, 64, 64),
            dtype="float32",
            timestamp=time.time(),
            source="any_source",  # Accepts any string currently
        )
        assert message.source == "any_source"


@pytest.mark.unit
class TestPromptMessage:
    """Test PromptMessage functionality"""

    def test_prompt_message_creation(self):
        """Test creating a PromptMessage"""
        prompt = "a beautiful landscape"
        timestamp = time.time()

        message = PromptMessage(prompt=prompt, timestamp=timestamp)

        assert message.tag == "prompt"
        assert message.prompt == prompt
        assert message.timestamp == timestamp

        logger.info(f"Created PromptMessage: '{prompt}'")

    def test_prompt_message_empty_prompt(self):
        """Test PromptMessage with empty prompt"""
        message = PromptMessage(prompt="", timestamp=time.time())

        assert message.prompt == ""
        assert message.tag == "prompt"


@pytest.mark.unit
class TestProcessedFrameMessage:
    """Test ProcessedFrameMessage functionality"""

    def test_processed_frame_message_creation(self, sample_frame_jpeg):
        """Test creating a ProcessedFrameMessage"""
        timestamp = time.time()
        processing_time = 0.123

        message = ProcessedFrameMessage(
            frame=encode_bytes(sample_frame_jpeg), timestamp=timestamp, processing_time=processing_time
        )

        assert message.tag == "processed_frame"
        assert message.timestamp == timestamp
        assert message.processing_time == processing_time

        # Verify frame data
        decoded_frame = decode_bytes(message.frame)
        assert decoded_frame == sample_frame_jpeg

        logger.info(f"Created ProcessedFrameMessage with {processing_time}s processing time")


@pytest.mark.unit
class TestCarrierMessage:
    """Test CarrierMessage functionality"""

    def test_carrier_message_with_frame(self, frame_message):
        """Test CarrierMessage containing FrameMessage"""
        carrier = CarrierMessage(content=frame_message)

        assert isinstance(carrier.content, FrameMessage)
        assert carrier.content.tag == "frame"
        assert carrier.content == frame_message

        logger.info("Successfully created CarrierMessage with FrameMessage")

    def test_carrier_message_with_latents(self, latents_message):
        """Test CarrierMessage containing LatentsMessage"""
        carrier = CarrierMessage(content=latents_message)

        assert isinstance(carrier.content, LatentsMessage)
        assert carrier.content.tag == "latents"
        assert carrier.content == latents_message

        logger.info("Successfully created CarrierMessage with LatentsMessage")

    def test_carrier_message_with_prompt(self, prompt_message):
        """Test CarrierMessage containing PromptMessage"""
        carrier = CarrierMessage(content=prompt_message)

        assert isinstance(carrier.content, PromptMessage)
        assert carrier.content.tag == "prompt"
        assert carrier.content == prompt_message

        logger.info("Successfully created CarrierMessage with PromptMessage")

    def test_carrier_message_serialization(self, frame_message):
        """Test CarrierMessage JSON serialization"""
        carrier = CarrierMessage(content=frame_message)

        # Test JSON serialization
        json_data = carrier.model_dump()
        assert isinstance(json_data, dict)
        assert "content" in json_data

        # Test deserialization
        reconstructed = CarrierMessage.model_validate(json_data)
        assert isinstance(reconstructed.content, FrameMessage)
        assert reconstructed.content.frame == frame_message.frame

        logger.info("Successfully serialized/deserialized CarrierMessage")


@pytest.mark.unit
class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_decode_invalid_base64(self):
        """Test decoding invalid base64 data"""
        with pytest.raises(Exception):  # Could be various exception types
            decode_bytes("invalid_base64!")

    def test_deserialize_invalid_array_data(self):
        """Test deserializing invalid array data"""
        with pytest.raises(Exception):
            deserialize_array("invalid_data", (10, 10), "float32")

    def test_deserialize_mismatched_shape(self):
        """Test deserializing with wrong shape"""
        original = np.array([1, 2, 3, 4], dtype=np.float32)
        data_str, _, dtype_str = serialize_array(original)

        # Try to deserialize with wrong shape
        with pytest.raises(ValueError):
            deserialize_array(data_str, (2, 3), dtype_str)  # Wrong shape

    def test_message_validation_errors(self):
        """Test message validation failures"""
        # Test missing required fields
        with pytest.raises(Exception):
            FrameMessage()  # Missing required fields

        # Test invalid timestamp
        with pytest.raises(Exception):
            FrameMessage(
                frame="test",
                timestamp="invalid_timestamp",  # Should be float
            )


@pytest.mark.unit
class TestPerformance:
    """Test performance characteristics"""

    def test_large_array_serialization_performance(self, performance_monitor):
        """Test performance with large arrays"""
        # Create a large array similar to high-res latents
        large_array = np.random.randn(4, 8, 128, 128).astype(np.float32)

        performance_monitor.start("large_array_serialization")

        # Serialize
        data_str, shape, dtype_str = serialize_array(large_array)

        # Deserialize
        reconstructed = deserialize_array(data_str, shape, dtype_str)

        duration = performance_monitor.end("large_array_serialization")

        # Verify correctness
        np.testing.assert_array_equal(reconstructed, large_array)

        # Performance assertion (should be reasonably fast)
        assert duration < 1.0, f"Large array serialization took too long: {duration}s"

        logger.info(f"Large array serialization: {duration:.3f}s for {large_array.nbytes} bytes")

    def test_message_creation_performance(self, sample_frame_jpeg, performance_monitor):
        """Test message creation performance"""
        performance_monitor.start("message_creation")

        # Create multiple messages
        for _ in range(100):
            message = FrameMessage(frame=encode_bytes(sample_frame_jpeg), timestamp=time.time(), camera_id=0)
            carrier = CarrierMessage(content=message)

        duration = performance_monitor.end("message_creation")

        # Should be very fast
        assert duration < 0.5, f"Message creation too slow: {duration}s for 100 messages"

        logger.info(f"Created 100 messages in {duration:.3f}s")
