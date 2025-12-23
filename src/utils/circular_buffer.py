"""Thread-safe circular buffer for frame storage."""
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class Frame:
    """Container for a captured frame with metadata."""
    image: np.ndarray
    timestamp: float
    frame_number: int = 0


class CircularBuffer:
    """Thread-safe circular buffer for storing recent frames.
    
    This buffer maintains a fixed-size queue of frames, automatically
    discarding the oldest frames when capacity is reached.
    """
    
    def __init__(self, maxsize: int = 30):
        """Initialize the circular buffer.
        
        Args:
            maxsize: Maximum number of frames to store
        """
        self.maxsize = maxsize
        self._buffer = deque(maxlen=maxsize)
        self._lock = threading.Lock()
        self._frame_count = 0
    
    def push(self, frame: Frame) -> None:
        """Add a frame to the buffer.
        
        Args:
            frame: Frame to add to the buffer
        """
        with self._lock:
            frame.frame_number = self._frame_count
            self._buffer.append(frame)
            self._frame_count += 1
    
    def peek_latest(self) -> Optional[Frame]:
        """Get the most recent frame without removing it.
        
        Returns:
            Most recent frame, or None if buffer is empty
        """
        with self._lock:
            if len(self._buffer) == 0:
                return None
            return self._buffer[-1]
    
    def pop_latest(self) -> Optional[Frame]:
        """Get and remove the most recent frame.
        
        Returns:
            Most recent frame, or None if buffer is empty
        """
        with self._lock:
            if len(self._buffer) == 0:
                return None
            return self._buffer.pop()
    
    def clear(self) -> None:
        """Clear all frames from the buffer."""
        with self._lock:
            self._buffer.clear()
    
    def __len__(self) -> int:
        """Get current number of frames in buffer."""
        with self._lock:
            return len(self._buffer)
    
    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self) == 0
    
    @property
    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        return len(self) >= self.maxsize

