from collections import deque

import librosa
import numpy as np


class AudioBuffer:
    """Rolling audio buffer for accumulating audio chunks."""

    def __init__(self, max_seconds: int, sample_rate: int = 16000):
        """Initialize audio buffer.

        Args:
            max_seconds: Maximum number of seconds to buffer
            sample_rate: Target sample rate (16kHz for Gemma 3n)
        """
        self.max_seconds = max_seconds
        self.sample_rate = sample_rate
        self.max_samples = max_seconds * sample_rate
        self.buffer: deque[np.ndarray] = deque()
        self.total_samples = 0

    def add_chunk(self, audio_data: bytes, source_sample_rate: int, channels: int) -> None:
        """Add an audio chunk to the buffer.

        Args:
            audio_data: PCM float32 audio data
            source_sample_rate: Sample rate of the incoming audio
            channels: Number of channels (1=mono, 2=stereo)
        """
        # Convert bytes to numpy array (float32)
        audio = np.frombuffer(audio_data, dtype=np.float32)

        # Convert to mono if stereo
        if channels == 2:
            audio = audio.reshape(-1, 2).mean(axis=1)

        # Resample to 16kHz if needed
        if source_sample_rate != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=source_sample_rate, target_sr=self.sample_rate)

        # Ensure audio is in [-1, 1] range
        audio = np.clip(audio, -1.0, 1.0)

        # Add to buffer
        self.buffer.append(audio)
        self.total_samples += len(audio)

        # Remove old chunks if we exceed max buffer size
        while self.total_samples > self.max_samples:
            old_chunk = self.buffer.popleft()
            self.total_samples -= len(old_chunk)

    def get_audio(self) -> np.ndarray:
        """Get the buffered audio as a single numpy array.

        Returns:
            Concatenated audio buffer as float32 array in [-1, 1]
        """
        if not self.buffer:
            return np.array([], dtype=np.float32)
        return np.concatenate(list(self.buffer))

    def clear(self) -> None:
        self.buffer.clear()
        self.total_samples = 0

    def partial_clear(self, keep_seconds: float = 5.0) -> None:
        keep_samples = int(keep_seconds * self.sample_rate)
        if self.total_samples <= keep_samples:
            return

        audio = self.get_audio()
        self.buffer.clear()
        kept_audio = audio[-keep_samples:]
        self.buffer.append(kept_audio)
        self.total_samples = len(kept_audio)

    def duration_seconds(self) -> float:
        """Get the current buffer duration in seconds."""
        return self.total_samples / self.sample_rate
