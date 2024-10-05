import cv2
import logging
import numpy as np
from typing import Tuple, Optional, List
from openvino.runtime import Core
import tensorflow as tf
import threading
from queue import Queue
import os

# Configure TensorFlow to use oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self, source_path: str):
        """
        Initialize video processor for bus number recognition
        Args:
            source_path (str): Path to the video file
        """
        self.cap = cv2.VideoCapture(source_path)
        if not self.cap.isOpened():
            logger.error("Failed to open video file")
            raise ValueError(f"Could not open video file: {source_path}")

        # Video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        # Initialize OpenVINO
        self.core = Core()

        # Load and optimize the model for bus number recognition
        self._initialize_model()

        # Frame processing queue
        self.frame_queue = Queue(maxsize=30)
        self.result_queue = Queue()
        self.processing_thread = threading.Thread(target=self._process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        logger.info(f"Video Processor initialized: {self.frame_width}x{self.frame_height} @ {self.fps}fps")

    def _initialize_model(self):
        """Initialize and optimize the bus number recognition model"""
        # Note: You'll need to replace this with your actual model path and architecture
        model_path = r"C:\Users\LAKSHMINARAYANAN\open_model_zoo\tools\model_tools\intel\vehicle-license-plate-detection-barrier-0106\FP16\vehicle-license-plate-detection-barrier-0106.xml"

        self.model = self.core.read_model(model_path)
        self.compiled_model = self.core.compile_model(self.model, "CPU")
        self.infer_request = self.compiled_model.create_infer_request()
        logger.info("Model loaded and optimized with OpenVINO")

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame using Intel MKL optimizations"""
        # Convert to float32 for better performance with MKL
        frame_float = frame.astype(np.float32)

        # Example preprocessing steps optimized with MKL
        # Resize using OpenCV with Intel optimizations
        resized_frame = cv2.resize(frame_float, (224, 224))

        # Normalize using numpy (automatically uses MKL)
        normalized_frame = (resized_frame - 128) / 128

        return normalized_frame

    def _process_frames(self):
        """Process frames in a separate thread"""
        while True:
            frame = self.frame_queue.get()
            if frame is None:
                break

            try:
                # Preprocess frame
                processed_frame = self._preprocess_frame(frame)

                # Perform inference using OpenVINO
                self.infer_request.set_tensor("input", processed_frame)
                self.infer_request.infer()
                result = self.infer_request.get_tensor("output").data

                # Post-process results (replace with your actual post-processing)
                bus_number = self._postprocess_result(result)

                self.result_queue.put(bus_number)
            except Exception as e:
                logger.error(f"Error processing frame: {str(e)}")
                self.result_queue.put(None)

    def _postprocess_result(self, result: np.ndarray) -> str:
        """Post-process the model output to get bus number"""
        # Replace with your actual post-processing logic
        return "Bus 123"  # Placeholder

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], Optional[str]]:
        """Read a frame from the video and return with processed results"""
        success, frame = self.cap.read()
        if not success:
            logger.warning("Failed to read frame")
            return False, None, None

        # Add frame to processing queue
        if not self.frame_queue.full():
            self.frame_queue.put(frame)
        else:
            logger.warning("Frame queue is full, skipping frame")

        # Get processed result if available
        bus_number = None
        if not self.result_queue.empty():
            bus_number = self.result_queue.get()

        return True, frame, bus_number

    def release(self):
        """Release resources"""
        self.frame_queue.put(None)  # Signal processing thread to stop
        self.processing_thread.join()
        self.cap.release()
        logger.info("Video processor released")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
