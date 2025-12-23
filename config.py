"""Configuration management for Real-Time Action Recognition system."""
from dataclasses import dataclass, field
from typing import Optional
import yaml
from pathlib import Path


@dataclass
class Config:
    """Configuration for the action recognition system."""
    
    # Camera settings
    camera_id: int = 0
    capture_width: int = 1280
    capture_height: int = 720
    capture_fps: int = 30
    buffer_size: int = 30
    
    # Inference settings
    target_fps: float = 2.0
    model_name: str = "vikhyatk/moondream2"
    model_type: str = "moondream"  # moondream or florence
    quantize: bool = True
    compile_model: bool = True
    max_tokens: int = 50
    
    # Change detection
    change_method: str = "mse"  # mse, ssim, or hash
    change_threshold: float = 0.15
    
    # Display settings
    display_mode: str = "window"  # overlay or window
    window_opacity: float = 0.85
    show_latency: bool = True
    show_fps: bool = True
    
    # Prompt settings
    prompt: str = "Describe the person's action, emotion, and environment in one sentence."
    
    # Voice Output (TTS)
    tts_enabled: bool = False
    tts_rate: int = 150
    tts_volume: float = 0.8
    
    # Video Recording
    recording_fps: int = 30
    recording_dir: str = "recordings"
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_dir: str = "logs"
    
    @classmethod
    def from_file(cls, path: str) -> 'Config':
        """Load configuration from YAML file.
        
        Args:
            path: Path to YAML configuration file
            
        Returns:
            Config instance with loaded settings
        """
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)
    
    def to_file(self, path: str) -> None:
        """Save configuration to YAML file.
        
        Args:
            path: Path to save YAML configuration file
        """
        config_dict = {
            'camera_id': self.camera_id,
            'capture_width': self.capture_width,
            'capture_height': self.capture_height,
            'capture_fps': self.capture_fps,
            'buffer_size': self.buffer_size,
            'target_fps': self.target_fps,
            'model_name': self.model_name,
            'quantize': self.quantize,
            'compile_model': self.compile_model,
            'max_tokens': self.max_tokens,
            'change_method': self.change_method,
            'change_threshold': self.change_threshold,
            'display_mode': self.display_mode,
            'window_opacity': self.window_opacity,
            'show_latency': self.show_latency,
            'show_fps': self.show_fps,
            'prompt': self.prompt,
            'log_level': self.log_level,
            'log_to_file': self.log_to_file,
            'log_dir': self.log_dir,
        }
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

