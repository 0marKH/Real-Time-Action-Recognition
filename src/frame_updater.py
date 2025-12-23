"""Separate thread for updating display with camera frames at high FPS."""
import time
import threading
from typing import Optional
from .camera import CameraCapture
from .display import OverlayDisplay
from .utils.logging_config import get_logger

logger = get_logger("frame_updater")


class FrameUpdater:
    """Updates display with camera frames at high frequency.
    
    Runs in its own thread to ensure smooth camera feed display
    independent of inference speed.
    """
    
    def __init__(self, camera: CameraCapture, display: OverlayDisplay, target_fps: int = 30, recorder=None):
        """Initialize frame updater.
        
        Args:
            camera: Camera capture instance
            display: Display instance
            target_fps: Target frames per second for display
            recorder: Optional video recorder instance
        """
        self.camera = camera
        self.display = display
        self.recorder = recorder
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        
        self.running = False
        self.thread: Optional[threading.Thread] = None
    
    def set_recorder(self, recorder) -> None:
        """Set the video recorder instance.
        
        Args:
            recorder: VideoRecorder instance
        """
        self.recorder = recorder
    
    def _update_loop(self) -> None:
        """Main update loop."""
        logger.info(f"Frame updater started (target: {self.target_fps} FPS)")
        
        while self.running:
            try:
                start_time = time.time()
                
                # Get latest frame from camera
                frame_obj = self.camera.get_latest_frame()
                if frame_obj is not None:
                    # Update display
                    self.display.update_frame(frame_obj.image)
                    
                    # Write to video recorder if recording
                    if self.recorder and self.recorder.is_recording():
                        self.recorder.write_frame(frame_obj.image)
                
                # Maintain target FPS
                elapsed = time.time() - start_time
                sleep_time = self.frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    time.sleep(0.001)  # Yield to other threads
            
            except Exception as e:
                logger.error(f"Frame update error: {e}")
                time.sleep(0.01)
        
        logger.info("Frame updater stopped")
    
    def start(self) -> None:
        """Start the frame updater thread."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(
            target=self._update_loop,
            daemon=True,
            name="FrameUpdater"
        )
        self.thread.start()
    
    def stop(self) -> None:
        """Stop the frame updater thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

