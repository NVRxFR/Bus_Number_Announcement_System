import numpy as np
import logging
from typing import List, Tuple, Optional
import re
import cv2
from openvino.runtime import Core, Tensor
import threading
from queue import Queue
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
import os

# Configure TensorFlow to use oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedTextDetector:
    def __init__(self, max_batch_size: int = 4):
        """
        Initialize the optimized text detector using Intel libraries
        Args:
            max_batch_size (int): Maximum batch size for parallel processing
        """
        logger.info("Initializing OptimizedTextDetector...")

        # Initialize OpenVINO
        self.core = Core()
        self._initialize_models()

        # Bus number pattern
        self.bus_number_pattern = re.compile(r'^\d{2,3}[A-Z]?$')

        # Batch processing setup
        self.max_batch_size = max_batch_size
        self.frame_queue = Queue(maxsize=max_batch_size * 2)
        self.result_queue = Queue()

        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count())

        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_batch)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        logger.info("OptimizedTextDetector initialized successfully")

    def _initialize_models(self):
        """Initialize and optimize detection and recognition models"""
        # Load text detection model
        det_model_path = r"C:\Users\LAKSHMINARAYANAN\open_model_zoo\tools\model_tools\intel\text-detection-0003\FP16\text-detection-0003.xml"
        self.det_model = self.core.read_model(det_model_path)
        self.compiled_det_model = self.core.compile_model(self.det_model, "CPU")

        # Load text recognition model
        rec_model_path = r"C:\Users\LAKSHMINARAYANAN\open_model_zoo\tools\model_tools\intel\text-recognition-0012\FP16\text-recognition-0012.xml"
        self.rec_model = self.core.read_model(rec_model_path)
        self.compiled_rec_model = self.core.compile_model(self.rec_model, "CPU")

        # Create inference requests
        self.det_infer_request = self.compiled_det_model.create_infer_request()
        self.rec_infer_request = self.compiled_rec_model.create_infer_request()

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame using Intel MKL optimizations"""
        # Convert to float32 for better performance with MKL
        frame_float = frame.astype(np.float32)

        # Resize and normalize
        resized_frame = cv2.resize(frame_float, (1024, 768))
        normalized_frame = (resized_frame - 127.5) / 127.5

        return normalized_frame

    def _detect_text_regions(self, preprocessed_frame: np.ndarray) -> List[np.ndarray]:
        """Detect text regions using OpenVINO-optimized model"""
        self.det_infer_request.set_tensor("input", preprocessed_frame)
        self.det_infer_request.infer()
        detection_results = self.det_infer_request.get_tensor("output").data

        # Post-process detection results to get text regions
        # Replace with actual post-processing logic
        text_regions = []  # Placeholder
        return text_regions

    def _recognize_text(self, text_regions: List[np.ndarray]) -> List[Tuple[str, float]]:
        """Recognize text in detected regions using OpenVINO-optimized model"""
        results = []
        for region in text_regions:
            self.rec_infer_request.set_tensor("input", region)
            self.rec_infer_request.infer()
            recognition_result = self.rec_infer_request.get_tensor("output").data

            # Post-process recognition results
            # Replace with actual post-processing logic
            text = "123"  # Placeholder
            confidence = 0.95  # Placeholder

            if self.is_valid_bus_number(text):
                results.append((text, confidence))

        return results

    def _process_batch(self):
        """Process batches of frames in a separate thread"""
        while True:
            batch = []
            for _ in range(self.max_batch_size):
                frame = self.frame_queue.get()
                if frame is None:
                    # End signal received
                    if batch:
                        self._process_frames(batch)
                    return
                batch.append(frame)
                if len(batch) == self.max_batch_size or self.frame_queue.empty():
                    break

            if batch:
                self._process_frames(batch)

    def _process_frames(self, frames: List[np.ndarray]):
        """Process a batch of frames in parallel"""
        try:
            # Preprocess frames in parallel
            preprocessed_frames = list(self.executor.map(self._preprocess_frame, frames))

            # Detect text regions in parallel
            text_regions_batch = list(self.executor.map(self._detect_text_regions, preprocessed_frames))

            # Recognize text in parallel
            results_batch = list(self.executor.map(self._recognize_text, text_regions_batch))

            for results in results_batch:
                self.result_queue.put(results)

        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            for _ in frames:
                self.result_queue.put([])

    def is_valid_bus_number(self, text: str) -> bool:
        """Check if the detected text matches bus number pattern"""
        return bool(self.bus_number_pattern.match(text))

    def add_frame(self, frame: np.ndarray):
        """Add a frame to the processing queue"""
        if not self.frame_queue.full():
            self.frame_queue.put(frame)
        else:
            logger.warning("Frame queue is full, skipping frame")

    def get_results(self) -> Optional[List[Tuple[str, float]]]:
        """Get processed results if available"""
        if not self.result_queue.empty():
            return self.result_queue.get()
        return None

    def release(self):
        """Release resources"""
        self.frame_queue.put(None)  # Signal processing thread to stop
        self.processing_thread.join()
        self.executor.shutdown()
        logger.info("OptimizedTextDetector released")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
