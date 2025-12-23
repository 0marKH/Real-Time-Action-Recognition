"""Video recording with caption overlay."""
import cv2
import numpy as np
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from .utils.logging_config import get_logger

logger = get_logger("video_recorder")


class VideoRecorder:
    """Records video with text captions overlaid."""
    
    def __init__(self, output_dir: str = "recordings", fps: int = 30):
        """Initialize video recorder.
        
        Args:
            output_dir: Directory to save recordings
            fps: Frames per second for output video
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.fps = fps
        self.recording = False
        self.writer: Optional[cv2.VideoWriter] = None
        self.current_file: Optional[Path] = None
        self.current_text: str = ""
        
        # Description history for summary
        self.descriptions: list[tuple[float, str]] = []  # (timestamp, text)
        self.recording_start_time: float = 0
        
        # Video settings
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.frame_width = 1280
        self.frame_height = 720
    
    def start_recording(self, width: int = 1280, height: int = 720) -> bool:
        """Start recording video.
        
        Args:
            width: Video width
            height: Video height
            
        Returns:
            True if started successfully
        """
        if self.recording:
            logger.warning("Already recording")
            return False
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_file = self.output_dir / f"recording_{timestamp}.mp4"
        
        # Update dimensions
        self.frame_width = width
        self.frame_height = height
        
        # Create video writer
        self.writer = cv2.VideoWriter(
            str(self.current_file),
            self.fourcc,
            self.fps,
            (self.frame_width, self.frame_height)
        )
        
        if not self.writer.isOpened():
            logger.error("Failed to open video writer")
            return False
        
        self.recording = True
        self.descriptions = []  # Reset description history
        self.recording_start_time = time.time()
        logger.info(f"Recording started: {self.current_file}")
        return True
    
    def stop_recording(self) -> Optional[Path]:
        """Stop recording and save video.
        
        Returns:
            Path to saved video file, or None if not recording
        """
        if not self.recording:
            return None
        
        self.recording = False
        
        if self.writer:
            self.writer.release()
            self.writer = None
        
        logger.info(f"Recording stopped: {self.current_file}")
        saved_file = self.current_file
        self.current_file = None
        
        return saved_file
    
    def update_text(self, text: str) -> None:
        """Update the current caption text.
        
        Args:
            text: Text to display on frames
        """
        self.current_text = text
        
        # Store description with timestamp if recording
        if self.recording:
            relative_time = time.time() - self.recording_start_time
            self.descriptions.append((relative_time, text))
    
    def write_frame(self, frame: np.ndarray) -> None:
        """Write a frame with caption overlay.
        
        Args:
            frame: Frame to write (BGR format)
        """
        if not self.recording or self.writer is None:
            return
        
        # Resize frame if needed
        if frame.shape[1] != self.frame_width or frame.shape[0] != self.frame_height:
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        
        # Add text overlay
        frame_with_text = self._add_text_overlay(frame.copy())
        
        # Write to video
        self.writer.write(frame_with_text)
    
    def _add_text_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Add text overlay to frame.
        
        Args:
            frame: Original frame
            
        Returns:
            Frame with text overlay
        """
        if not self.current_text:
            return frame
        
        # Create semi-transparent overlay box
        overlay = frame.copy()
        height, width = frame.shape[:2]
        
        # Calculate text size and position
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # Split text into multiple lines if too long
        max_width = width - 40
        words = self.current_text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
            
            if text_width < max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Calculate box dimensions
        line_height = 35
        box_height = len(lines) * line_height + 20
        box_y = height - box_height - 10
        
        # Draw semi-transparent box
        cv2.rectangle(
            overlay,
            (10, box_y),
            (width - 10, height - 10),
            (0, 0, 0),
            -1
        )
        
        # Blend with original
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw text lines
        for i, line in enumerate(lines):
            y_pos = box_y + 25 + (i * line_height)
            cv2.putText(
                frame,
                line,
                (20, y_pos),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )
        
        # Add recording indicator
        cv2.circle(frame, (width - 30, 30), 10, (0, 0, 255), -1)
        cv2.putText(
            frame,
            "REC",
            (width - 70, 35),
            font,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )
        
        return frame
    
    def is_recording(self) -> bool:
        """Check if currently recording.
        
        Returns:
            True if recording
        """
        return self.recording
    
    def generate_summary(self) -> str:
        """Generate a summary of the recording from descriptions.
        
        Returns:
            Summary text
        """
        if not self.descriptions:
            return "No descriptions recorded."
        
        # Calculate recording duration
        duration = self.descriptions[-1][0] if self.descriptions else 0
        duration_str = f"{int(duration // 60)}:{int(duration % 60):02d}"
        
        # Create a narrative summary
        summary_parts = []
        summary_parts.append(f"Video Summary ({duration_str})")
        summary_parts.append("=" * 50)
        summary_parts.append("")
        
        # Group similar consecutive descriptions
        grouped = []
        last_desc = None
        count = 0
        
        for timestamp, desc in self.descriptions:
            if desc == last_desc:
                count += 1
            else:
                if last_desc:
                    time_str = f"{int(timestamp // 60)}:{int(timestamp % 60):02d}"
                    if count > 1:
                        grouped.append(f"[{time_str}] {last_desc} (repeated {count}x)")
                    else:
                        grouped.append(f"[{time_str}] {last_desc}")
                last_desc = desc
                count = 1
        
        # Add the last one
        if last_desc:
            timestamp = self.descriptions[-1][0]
            time_str = f"{int(timestamp // 60)}:{int(timestamp % 60):02d}"
            if count > 1:
                grouped.append(f"[{time_str}] {last_desc} (repeated {count}x)")
            else:
                grouped.append(f"[{time_str}] {last_desc}")
        
        # Add timeline
        summary_parts.append("Timeline:")
        summary_parts.append("-" * 50)
        for item in grouped[:20]:  # Show first 20 unique descriptions
            summary_parts.append(item)
        
        if len(grouped) > 20:
            summary_parts.append(f"... and {len(grouped) - 20} more events")
        
        summary_parts.append("")
        summary_parts.append(f"Total unique events: {len(grouped)}")
        summary_parts.append(f"Total descriptions: {len(self.descriptions)}")
        
        return "\n".join(summary_parts)
    
    def save_summary(self) -> Optional[Path]:
        """Save the recording summary to a text file.
        
        Returns:
            Path to saved summary file, or None if no recording
        """
        if not self.current_file:
            return None
        
        summary = self.generate_summary()
        summary_file = self.current_file.with_suffix('.txt')
        
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(summary)
            
            logger.info(f"Summary saved: {summary_file}")
            return summary_file
        except Exception as e:
            logger.error(f"Failed to save summary: {e}")
            return None

