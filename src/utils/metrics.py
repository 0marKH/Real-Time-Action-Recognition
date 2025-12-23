"""Performance metrics tracking for the action recognition system."""
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional
import threading


@dataclass
class Description:
    """Container for a generated description with metadata."""
    text: str
    timestamp: float
    latency: float
    frame_number: int = 0


class MetricsCollector:
    """Collects and tracks performance metrics.
    
    Tracks inference latency, FPS, and other performance indicators
    using a rolling window approach.
    """
    
    def __init__(self, window_size: int = 30):
        """Initialize metrics collector.
        
        Args:
            window_size: Number of samples to keep for rolling averages
        """
        self.window_size = window_size
        self._inference_times = deque(maxlen=window_size)
        self._fps_samples = deque(maxlen=window_size)
        self._last_frame_time = None
        self._lock = threading.Lock()
        
        # Counters
        self.total_inferences = 0
        self.total_frames = 0
        self.skipped_frames = 0
        self.start_time = time.time()
    
    def record_inference(self, latency: float) -> None:
        """Record an inference timing.
        
        Args:
            latency: Time taken for inference in seconds
        """
        with self._lock:
            self._inference_times.append(latency)
            self.total_inferences += 1
    
    def record_frame(self, skipped: bool = False) -> None:
        """Record a frame being processed or skipped.
        
        Args:
            skipped: Whether the frame was skipped (not processed)
        """
        with self._lock:
            current_time = time.time()
            
            if self._last_frame_time is not None:
                frame_delta = current_time - self._last_frame_time
                if frame_delta > 0:
                    self._fps_samples.append(1.0 / frame_delta)
            
            self._last_frame_time = current_time
            self.total_frames += 1
            
            if skipped:
                self.skipped_frames += 1
    
    @property
    def avg_inference_time(self) -> float:
        """Get average inference time in milliseconds."""
        with self._lock:
            if not self._inference_times:
                return 0.0
            return sum(self._inference_times) / len(self._inference_times) * 1000
    
    @property
    def latest_inference_time(self) -> float:
        """Get most recent inference time in milliseconds."""
        with self._lock:
            if not self._inference_times:
                return 0.0
            return self._inference_times[-1] * 1000
    
    @property
    def avg_fps(self) -> float:
        """Get average FPS from recent samples."""
        with self._lock:
            if not self._fps_samples:
                return 0.0
            return sum(self._fps_samples) / len(self._fps_samples)
    
    @property
    def current_fps(self) -> float:
        """Get instantaneous FPS."""
        with self._lock:
            if not self._fps_samples:
                return 0.0
            return self._fps_samples[-1]
    
    @property
    def skip_rate(self) -> float:
        """Get percentage of frames skipped."""
        with self._lock:
            if self.total_frames == 0:
                return 0.0
            return (self.skipped_frames / self.total_frames) * 100
    
    @property
    def uptime(self) -> float:
        """Get system uptime in seconds."""
        return time.time() - self.start_time
    
    def get_summary(self) -> dict:
        """Get a summary of all metrics.
        
        Returns:
            Dictionary containing all current metrics
        """
        with self._lock:
            return {
                'avg_inference_ms': self.avg_inference_time,
                'latest_inference_ms': self.latest_inference_time,
                'avg_fps': self.avg_fps,
                'current_fps': self.current_fps,
                'total_inferences': self.total_inferences,
                'total_frames': self.total_frames,
                'skipped_frames': self.skipped_frames,
                'skip_rate': self.skip_rate,
                'uptime_seconds': self.uptime,
            }
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._inference_times.clear()
            self._fps_samples.clear()
            self._last_frame_time = None
            self.total_inferences = 0
            self.total_frames = 0
            self.skipped_frames = 0
            self.start_time = time.time()

