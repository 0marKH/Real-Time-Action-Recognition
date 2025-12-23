"""Camera capture module with threading support."""
import cv2
import time
import threading
from typing import Optional
from .utils.circular_buffer import CircularBuffer, Frame
from .utils.logging_config import get_logger

logger = get_logger("camera")


class CameraCapture:
    """Threaded camera capture with circular buffer.
    
    Captures frames continuously in a separate thread and stores them
    in a circular buffer for non-blocking access.
    """
    
    def __init__(
        self,
        camera_id: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        buffer_size: int = 30
    ):
        """Initialize camera capture.
        
        Args:
            camera_id: Camera device index (0 for default camera)
            width: Capture resolution width
            height: Capture resolution height
            fps: Target capture FPS
            buffer_size: Number of frames to keep in buffer
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = buffer_size
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.buffer = CircularBuffer(maxsize=buffer_size)
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Error tracking
        self.error_count = 0
        self.max_retries = 3
        self.last_error: Optional[str] = None
    
    def _initialize_camera(self) -> bool:
        """Initialize the camera with specified settings.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Initializing camera {self.camera_id}...")
            
            # Try to open camera
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_id}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
            
            # Verify settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            logger.info(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
            
            # Warm up camera (read a few frames)
            for _ in range(5):
                ret, _ = self.cap.read()
                if not ret:
                    logger.warning("Camera warm-up frame failed")
            
            self.error_count = 0
            self.last_error = None
            return True
            
        except Exception as e:
            logger.error(f"Camera initialization error: {e}")
            self.last_error = str(e)
            return False
    
    def _capture_loop(self) -> None:
        """Main capture loop running in separate thread."""
        logger.info("Capture thread started")
        consecutive_failures = 0
        
        while self.running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    logger.warning("Camera disconnected, attempting to reconnect...")
                    if not self._initialize_camera():
                        consecutive_failures += 1
                        if consecutive_failures >= self.max_retries:
                            logger.error("Max reconnection attempts reached, stopping capture")
                            self.running = False
                            break
                        time.sleep(1.0 * consecutive_failures)  # Exponential backoff
                        continue
                    consecutive_failures = 0
                
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    consecutive_failures += 1
                    if consecutive_failures >= 10:
                        logger.error("Too many consecutive frame read failures")
                        self.cap.release()
                        self.cap = None
                        consecutive_failures = 0
                    time.sleep(0.01)
                    continue
                
                # Successfully read frame
                consecutive_failures = 0
                
                # Store in buffer
                self.buffer.push(Frame(
                    image=frame,
                    timestamp=time.time()
                ))
                
                # Small sleep to avoid spinning too fast
                time.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                consecutive_failures += 1
                time.sleep(0.1)
        
        logger.info("Capture thread stopped")
    
    def start(self) -> bool:
        """Start camera capture in background thread.
        
        Returns:
            True if started successfully, False otherwise
        """
        if self.running:
            logger.warning("Camera capture already running")
            return True
        
        # Initialize camera
        if not self._initialize_camera():
            logger.error("Failed to initialize camera")
            return False
        
        # Start capture thread
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        
        logger.info("Camera capture started")
        return True
    
    def stop(self) -> None:
        """Stop camera capture and release resources."""
        if not self.running:
            return
        
        logger.info("Stopping camera capture...")
        self.running = False
        
        # Wait for thread to finish
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        
        # Release camera
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        logger.info("Camera capture stopped")
    
    def get_latest_frame(self) -> Optional[Frame]:
        """Get the most recent frame from the buffer.
        
        Returns:
            Most recent frame, or None if no frames available
        """
        return self.buffer.peek_latest()
    
    def is_running(self) -> bool:
        """Check if capture is running.
        
        Returns:
            True if capture is active, False otherwise
        """
        return self.running
    
    def get_buffer_size(self) -> int:
        """Get current number of frames in buffer.
        
        Returns:
            Number of frames in buffer
        """
        return len(self.buffer)
    
    @staticmethod
    def list_available_cameras(max_cameras: int = 10) -> list[int]:
        """List available camera indices.
        
        Args:
            max_cameras: Maximum number of cameras to check
            
        Returns:
            List of available camera indices
        """
        available = []
        logger.info("Scanning for available cameras...")
        
        for i in range(max_cameras):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
                logger.info(f"Found camera at index {i}")
        
        return available
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

