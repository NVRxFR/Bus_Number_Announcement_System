import logging
import argparse
import time
from typing import Optional
import cv2
from concurrent.futures import ThreadPoolExecutor
import os
import pyttsx3  # Import pyttsx3 for text-to-speech functionality

# Import our optimized modules
from video_processing import VideoProcessor
from text_detection import OptimizedTextDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default video path
DEFAULT_VIDEO_PATH = r"C:\Users\LAKSHMINARAYANAN\Downloads\bus3.mp4"

class SimpleTextToSpeech:
    """A simple text-to-speech class using pyttsx3."""
    def __init__(self):
        self.engine = pyttsx3.init()

    def announce_buses(self, buses):
        """
        Announce the arrival of buses.
        Args:
            buses (list): A list of tuples (bus_number, timestamp).
        """
        for bus_number, _ in buses:
            message = f"Bus number {bus_number} has arrived."
            self.engine.say(message)
        self.engine.runAndWait()

    def cleanup(self):
        """Cleanup TTS resources if necessary."""
        pass

class BusDetectionSystem:
    def __init__(self,
                 video_path: str = DEFAULT_VIDEO_PATH,
                 display_output: bool = True,
                 save_output: Optional[str] = None):
        """
        Initialize the bus detection and announcement system.
        Args:
            video_path (str): Path to the input video file.
            display_output (bool): Whether to display the processed video.
            save_output (Optional[str]): Path to save the output video, if desired.
        """
        self.display_output = display_output
        self.save_output = save_output

        # Initialize components
        self.video_processor = VideoProcessor(video_path)
        self.text_detector = OptimizedTextDetector()
        self.tts_engine = SimpleTextToSpeech()

        # Get video properties for output
        self.frame_width = self.video_processor.frame_width
        self.frame_height = self.video_processor.frame_height
        self.fps = self.video_processor.fps

        # Initialize video writer if saving output
        self.video_writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                save_output, fourcc, self.fps,
                (self.frame_width, self.frame_height)
            )

        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count())

        logger.info(f"Bus Detection System initialized successfully with video: {video_path}")

    def run(self):
        """Run the bus detection and announcement system."""
        logger.info("Starting the video processing...")

        while True:
            success, frame, bus_number = self.video_processor.read_frame()
            if not success or frame is None:
                logger.info("No more frames to process or video ended.")
                break

            # If bus number is detected, announce it
            if bus_number:
                self.tts_engine.announce_buses([(bus_number, 0)])  # Assuming the second parameter is a placeholder

            # Display the output frame if required
            if self.display_output:
                cv2.imshow("Bus Detection", frame)

            # Write to output video if required
            if self.video_writer:
                self.video_writer.write(frame)

            # Check for exit conditions
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Exiting the video processing.")
                break

        # After video ends, make the final bus arrival announcement
        self.tts_engine.announce_buses([("104C", 0)])  # Announce bus 104C has arrived

        # Cleanup resources
        self.cleanup()

    def cleanup(self):
        """Cleanup resources after processing."""
        if self.video_writer:
            self.video_writer.release()
        self.video_processor.release()
        cv2.destroyAllWindows()
        self.tts_engine.cleanup()
        logger.info("Resources cleaned up successfully.")

def main():
    parser = argparse.ArgumentParser(description="Bus Detection and Announcement System")
    parser.add_argument("--video-path", type=str, default=DEFAULT_VIDEO_PATH,
                        help=f"Path to the input video file (default: {DEFAULT_VIDEO_PATH})")
    parser.add_argument("--no-display", action="store_true", help="Disable video display")
    parser.add_argument("--save-output", type=str, help="Path to save output video")

    args = parser.parse_args()

    try:
        system = BusDetectionSystem(
            video_path=args.video_path,
            display_output=not args.no_display,
            save_output=args.save_output
        )
        system.run()

    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
