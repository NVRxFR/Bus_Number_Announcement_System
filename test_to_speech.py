import logging
from threading import Thread, Event
from queue import Queue
from typing import List, Tuple
import numpy as np
import os
import sounddevice as sd
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleTextToSpeech:
    def __init__(self,
                 sample_rate: int = 22050,
                 max_batch_size: int = 4):
        """
        Initialize a simple text-to-speech engine.
        Args:
            sample_rate (int): Audio sample rate.
            max_batch_size (int): Maximum batch size for processing.
        """
        logger.info("Initializing SimpleTextToSpeech...")

        # Audio settings
        self.sample_rate = sample_rate

        # Batch processing setup
        self.max_batch_size = max_batch_size
        self.text_queue = Queue(maxsize=max_batch_size * 2)
        self.audio_queue = Queue()

        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count())

        # Thread management
        self.stop_event = Event()
        self.processing_thread = Thread(target=self._process_batch, daemon=True)
        self.playback_thread = Thread(target=self._playback_worker, daemon=True)

        # Start threads
        self.processing_thread.start()
        self.playback_thread.start()

        logger.info("SimpleTextToSpeech initialized successfully")

    def _text_to_phonemes(self, text: str) -> np.ndarray:
        """Convert text to a simple phoneme representation."""
        # Here we can use a basic representation
        phoneme_length = len(text)  # Length of text determines phoneme representation
        return np.random.rand(phoneme_length).astype(np.float32)  # Random float array simulates phonemes

    def _generate_audio(self, phonemes: np.ndarray) -> np.ndarray:
        """Generate synthetic audio from phoneme representation."""
        # Here we simulate audio generation from phonemes
        audio = np.sin(2 * np.pi * np.linspace(0, 1, 44100) * phonemes.sum())  # Example sine wave
        return audio

    def _create_announcement(self, bus_numbers: List[Tuple[str, float]]) -> str:
        """Create announcement text from bus numbers."""
        if len(bus_numbers) == 1:
            return f"Bus number {bus_numbers[0][0]} is approaching."
        else:
            bus_list = ", ".join([num for num, _ in bus_numbers[:-1]])
            return f"Bus numbers {bus_list} and {bus_numbers[-1][0]} are approaching."

    def _process_single_text(self, text: str) -> np.ndarray:
        """Process single text to audio."""
        try:
            phonemes = self._text_to_phonemes(text)
            audio = self._generate_audio(phonemes)
            return audio
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            return np.zeros((44100,), dtype=np.float32)  # Simulate silent audio

    def _process_batch(self):
        """Process batches of text in a separate thread."""
        while not self.stop_event.is_set():
            batch = []
            for _ in range(self.max_batch_size):
                try:
                    text = self.text_queue.get(timeout=1)
                    batch.append(text)
                except Exception:
                    break

                if len(batch) == self.max_batch_size:
                    break

            if batch:
                # Process batch in parallel
                audio_segments = list(self.executor.map(self._process_single_text, batch))
                for audio in audio_segments:
                    self.audio_queue.put(audio)

    def _playback_worker(self):
        """Worker thread for playing audio."""
        while not self.stop_event.is_set():
            try:
                audio = self.audio_queue.get(timeout=1)
                sd.play(audio, self.sample_rate)
                sd.wait()
            except Exception as e:
                logger.error(f"Error in playback: {str(e)}")

    def announce_buses(self, bus_numbers: List[Tuple[str, float]]):
        """Queue bus numbers for announcement."""
        if bus_numbers:
            announcement_text = self._create_announcement(bus_numbers)
            if not self.text_queue.full():
                self.text_queue.put(announcement_text)
            else:
                logger.warning("Announcement queue is full, skipping announcement.")

    def cleanup(self):
        """Cleanup resources."""
        self.stop_event.set()
        self.processing_thread.join()
        self.playback_thread.join()
        self.executor.shutdown()
        logger.info("SimpleTextToSpeech cleaned up.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


