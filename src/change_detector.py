"""Change detection for identifying significant frame changes."""
import cv2
import numpy as np
from typing import Optional
from .utils.logging_config import get_logger

logger = get_logger("change_detector")


class ChangeDetector:
    """Detects significant changes between consecutive frames.
    
    Uses various methods (MSE, SSIM, perceptual hashing) to determine
    if a new frame is different enough to warrant processing.
    """
    
    def __init__(
        self,
        method: str = "mse",
        threshold: float = 0.15,
        comparison_size: tuple[int, int] = (64, 64)
    ):
        """Initialize change detector.
        
        Args:
            method: Detection method ('mse', 'ssim', or 'hash')
            threshold: Threshold for detecting change (0-1, higher = less sensitive)
            comparison_size: Size to resize frames for comparison (smaller = faster)
        """
        self.method = method.lower()
        self.threshold = threshold
        self.comparison_size = comparison_size
        
        self.last_frame: Optional[np.ndarray] = None
        self._frame_count = 0
        self._change_count = 0
        
        # Validate method
        valid_methods = ["mse", "ssim", "hash"]
        if self.method not in valid_methods:
            logger.warning(f"Invalid method '{self.method}', using 'mse'")
            self.method = "mse"
        
        logger.info(f"Change detector initialized: method={self.method}, threshold={self.threshold}")
    
    def has_changed(self, frame: np.ndarray) -> bool:
        """Check if frame has changed significantly from last processed frame.
        
        Args:
            frame: Current frame (BGR format from OpenCV)
            
        Returns:
            True if frame has changed significantly, False otherwise
        """
        self._frame_count += 1
        
        # First frame always counts as changed
        if self.last_frame is None:
            self._update_reference(frame)
            self._change_count += 1
            return True
        
        # Detect change based on method
        changed = False
        
        if self.method == "mse":
            changed = self._detect_mse(frame)
        elif self.method == "ssim":
            changed = self._detect_ssim(frame)
        elif self.method == "hash":
            changed = self._detect_hash(frame)
        
        if changed:
            self._update_reference(frame)
            self._change_count += 1
        
        return changed
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for comparison.
        
        Args:
            frame: Input frame
            
        Returns:
            Preprocessed grayscale frame
        """
        # Resize for faster comparison
        small = cv2.resize(frame, self.comparison_size)
        
        # Convert to grayscale
        if len(small.shape) == 3:
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        else:
            gray = small
        
        return gray
    
    def _update_reference(self, frame: np.ndarray) -> None:
        """Update the reference frame.
        
        Args:
            frame: New reference frame
        """
        self.last_frame = self._preprocess_frame(frame)
    
    def _detect_mse(self, frame: np.ndarray) -> bool:
        """Detect change using Mean Squared Error.
        
        Args:
            frame: Current frame
            
        Returns:
            True if change detected
        """
        current = self._preprocess_frame(frame)
        
        # Calculate MSE
        diff = np.mean((current.astype(float) - self.last_frame.astype(float)) ** 2)
        diff_normalized = diff / (255 ** 2)  # Normalize to 0-1
        
        return diff_normalized > self.threshold
    
    def _detect_ssim(self, frame: np.ndarray) -> bool:
        """Detect change using Structural Similarity Index.
        
        Args:
            frame: Current frame
            
        Returns:
            True if change detected
        """
        try:
            from skimage.metrics import structural_similarity
        except ImportError:
            logger.warning("scikit-image not available, falling back to MSE")
            self.method = "mse"
            return self._detect_mse(frame)
        
        current = self._preprocess_frame(frame)
        
        # Calculate SSIM
        score = structural_similarity(current, self.last_frame)
        
        # SSIM ranges from -1 to 1, where 1 is identical
        # Convert to change score (0 = identical, 1 = completely different)
        change_score = 1 - score
        
        return change_score > self.threshold
    
    def _detect_hash(self, frame: np.ndarray) -> bool:
        """Detect change using perceptual hashing.
        
        Args:
            frame: Current frame
            
        Returns:
            True if change detected
        """
        current = self._preprocess_frame(frame)
        
        # Compute perceptual hashes
        hash1 = self._perceptual_hash(current)
        hash2 = self._perceptual_hash(self.last_frame)
        
        # Calculate Hamming distance
        hamming_dist = bin(hash1 ^ hash2).count('1') / 64.0
        
        return hamming_dist > self.threshold
    
    def _perceptual_hash(self, gray_image: np.ndarray) -> int:
        """Compute 64-bit perceptual hash of grayscale image.
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            64-bit hash as integer
        """
        # Resize to 8x8
        resized = cv2.resize(gray_image, (8, 8), interpolation=cv2.INTER_AREA)
        
        # Compute mean
        mean = resized.mean()
        
        # Generate hash (1 if pixel > mean, 0 otherwise)
        hash_bits = ''.join('1' if pixel > mean else '0' for pixel in resized.flatten())
        
        return int(hash_bits, 2)
    
    def reset(self) -> None:
        """Reset detector state."""
        self.last_frame = None
        self._frame_count = 0
        self._change_count = 0
        logger.info("Change detector reset")
    
    def get_stats(self) -> dict:
        """Get detector statistics.
        
        Returns:
            Dictionary with statistics
        """
        if self._frame_count == 0:
            change_rate = 0.0
        else:
            change_rate = (self._change_count / self._frame_count) * 100
        
        return {
            "method": self.method,
            "threshold": self.threshold,
            "total_frames": self._frame_count,
            "changes_detected": self._change_count,
            "change_rate_percent": change_rate,
        }
    
    def set_threshold(self, threshold: float) -> None:
        """Update detection threshold.
        
        Args:
            threshold: New threshold value (0-1)
        """
        if 0.0 <= threshold <= 1.0:
            self.threshold = threshold
            logger.info(f"Threshold updated to {threshold}")
        else:
            logger.warning(f"Invalid threshold {threshold}, must be between 0 and 1")

