import logging
import time
import asyncio
import numpy as np
from contextlib import AsyncExitStack
from faststream.redis import RedisBroker
from pydantic_settings import BaseSettings
from tenacity import retry, retry_if_exception_type, wait_exponential, stop_after_attempt

from mirror_mirror.common import log_errors
from mirror_mirror.models import AudioMessage, CarrierMessage

logger = logging.getLogger(__name__)


class Config(BaseSettings):
    redis_url: str = "redis://localhost:6379"
    sample_rate: int = 16000
    chunk_duration: float = 2.0  # seconds
    vad_threshold: float = 0.5  # voice activity threshold
    silence_timeout: float = 1.0  # seconds of silence before stopping


config = Config()


class AudioCapture:
    def __init__(self):
        self.sample_rate = config.sample_rate
        self.chunk_size = int(config.sample_rate * config.chunk_duration)
        self.audio_stream = None
        
    def initialize_audio(self):
        """Initialize audio capture"""
        try:
            import pyaudio
            self.audio = pyaudio.PyAudio()
            
            # Find the default input device
            device_info = self.audio.get_default_input_device_info()
            logger.info(f"Using audio device: {device_info['name']}")
            
            self.audio_stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=1024,
                input_device_index=device_info['index']
            )
            
            logger.info(f"Audio capture initialized: {self.sample_rate}Hz")
            
        except ImportError:
            logger.error("PyAudio not available. Install with: pip install pyaudio")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize audio: {e}")
            raise
    
    def cleanup(self):
        """Clean up audio resources"""
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        if hasattr(self, 'audio'):
            self.audio.terminate()
    
    def detect_voice_activity(self, audio_data: np.ndarray) -> bool:
        """Simple voice activity detection based on RMS energy"""
        rms = np.sqrt(np.mean(audio_data ** 2))
        return rms > config.vad_threshold
    
    def capture_audio_chunk(self) -> tuple[np.ndarray, bool]:
        """Capture a chunk of audio and detect voice activity"""
        if not self.audio_stream:
            self.initialize_audio()
        
        # Read audio data
        frames = []
        for _ in range(0, int(self.sample_rate / 1024 * config.chunk_duration)):
            data = self.audio_stream.read(1024, exception_on_overflow=False)
            frames.append(data)
        
        # Convert to numpy array
        audio_data = b''.join(frames)
        audio_array = np.frombuffer(audio_data, dtype=np.float32)
        
        # Detect voice activity
        has_voice = self.detect_voice_activity(audio_array)
        
        return audio_array, has_voice


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
        
        audio_capture = AudioCapture()
        stack.callback(audio_capture.cleanup)
        
        audio_capture.initialize_audio()
        
        logger.info("Starting audio capture with VAD...")
        
        silence_start = None
        recording = False
        
        while True:
            try:
                current_time = time.time()
                
                # Capture audio chunk
                audio_data, has_voice = audio_capture.capture_audio_chunk()
                
                if has_voice:
                    if not recording:
                        logger.info("Voice activity detected, starting recording")
                        recording = True
                    
                    silence_start = None
                    
                    # Send audio chunk
                    audio_bytes = audio_data.astype(np.float32).tobytes()
                    message = AudioMessage(
                        audio_data=audio_bytes,
                        sample_rate=config.sample_rate,
                        timestamp=current_time
                    )
                    
                    await broker.publish(
                        message=CarrierMessage(content=message),
                        channel="audio:chunks",
                    )
                    
                    logger.debug(f"Published audio chunk: {len(audio_data)} samples")
                    
                else:
                    if recording:
                        if silence_start is None:
                            silence_start = current_time
                        elif current_time - silence_start > config.silence_timeout:
                            logger.info("Silence detected, stopping recording")
                            recording = False
                            silence_start = None
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in audio capture: {e}")
                await asyncio.sleep(1.0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main()) 