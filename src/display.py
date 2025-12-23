"""Display module for showing camera feed and descriptions."""
import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
from typing import Optional, Callable
from .utils.metrics import Description, MetricsCollector
from .utils.logging_config import get_logger

logger = get_logger("display")


class OverlayDisplay:
    """Display window for camera feed and text descriptions.
    
    Supports two modes:
    - 'window': Full window with camera feed and text overlay
    - 'overlay': Transparent text-only overlay (always on top)
    """
    
    def __init__(
        self,
        mode: str = "window",
        opacity: float = 1.0,
        show_latency: bool = True,
        show_fps: bool = True
    ):
        """Initialize display.
        
        Args:
            mode: Display mode ('window' or 'overlay')
            opacity: Window opacity (0.0-1.0)
            show_latency: Whether to show latency indicator
            show_fps: Whether to show FPS counter
        """
        self.mode = mode
        self.opacity = opacity
        self.show_latency = show_latency
        self.show_fps = show_fps
        
        # UI elements
        self.root: Optional[tk.Tk] = None
        self.text_label: Optional[tk.Label] = None
        self.latency_label: Optional[tk.Label] = None
        self.fps_label: Optional[tk.Label] = None
        self.canvas: Optional[tk.Canvas] = None
        self.status_label: Optional[tk.Label] = None
        
        # State
        self.current_text = "Initializing..."
        self.current_latency = 0.0
        self.current_fps = 0.0
        self.current_frame: Optional[np.ndarray] = None
        self.paused = False
        self.running = False
        
        # Callbacks
        self.on_quit: Optional[Callable] = None
        self.on_pause: Optional[Callable] = None
        self.on_save: Optional[Callable] = None
        self.on_prompt_change: Optional[Callable] = None
        self.on_tts_toggle: Optional[Callable] = None
        self.on_record_toggle: Optional[Callable] = None
        self.on_custom_prompt: Optional[Callable] = None
        
        # Additional UI elements
        self.tts_button: Optional[tk.Button] = None
        self.record_button: Optional[tk.Button] = None
        self.prompt_entry: Optional[tk.Entry] = None
        
        # Update lock
        self._lock = threading.Lock()
        
        logger.info(f"Display initialized: mode={mode}, opacity={opacity}")
    
    def setup(self) -> None:
        """Set up the display window."""
        self.root = tk.Tk()
        self.root.title("Real-Time Action Recognition")
        
        # Set opacity only if less than 1.0
        if self.opacity < 1.0:
            self.root.attributes('-alpha', self.opacity)
        
        # Configure theme
        style = ttk.Style()
        style.theme_use('clam')
        
        # Modern dark theme colors
        bg_color = '#2b2b2b'  # Darker, more solid background
        fg_color = '#ffffff'
        accent_color = '#00d4aa'  # Brighter accent
        
        self.root.configure(bg=bg_color)
        
        # Set minimum window size
        self.root.minsize(680, 600)
        
        if self.mode == "window":
            self._setup_window_mode(bg_color, fg_color, accent_color)
        else:
            self._setup_overlay_mode(bg_color, fg_color, accent_color)
        
        # Bind keyboard shortcuts
        self._bind_keys()
        
        # Handle window close - force immediate shutdown
        self.root.protocol("WM_DELETE_WINDOW", self._on_window_close)
        
        # Bind Ctrl+C for force quit
        self.root.bind('<Control-c>', lambda e: self._on_quit())
        
        logger.info("Display setup complete")
    
    def _setup_window_mode(self, bg_color: str, fg_color: str, accent_color: str) -> None:
        """Set up full window with camera feed."""
        # Main container with padding
        main_frame = tk.Frame(self.root, bg=bg_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Camera feed canvas with border
        canvas_frame = tk.Frame(main_frame, bg='#1a1a1a', relief=tk.SUNKEN, borderwidth=2)
        canvas_frame.pack(pady=(0, 15))
        
        self.canvas = tk.Canvas(
            canvas_frame,
            bg='#000000',
            width=640,
            height=480,
            highlightthickness=0
        )
        self.canvas.pack(padx=2, pady=2)
        
        # Text description with styled frame
        text_frame = tk.Frame(main_frame, bg='#1a1a1a', relief=tk.FLAT, borderwidth=1)
        text_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.text_label = tk.Label(
            text_frame,
            text=self.current_text,
            font=('Segoe UI', 13),
            fg=fg_color,
            bg='#1a1a1a',
            wraplength=620,
            justify='left',
            padx=15,
            pady=15
        )
        self.text_label.pack()
        
        # Metrics row with styled frame
        metrics_frame = tk.Frame(main_frame, bg='#1a1a1a', relief=tk.FLAT, borderwidth=1)
        metrics_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Metrics container with padding
        metrics_inner = tk.Frame(metrics_frame, bg='#1a1a1a')
        metrics_inner.pack(fill=tk.X, padx=10, pady=8)
        
        if self.show_latency:
            self.latency_label = tk.Label(
                metrics_inner,
                text="⏱ Latency: --ms",
                font=('Segoe UI', 10, 'bold'),
                fg=accent_color,
                bg='#1a1a1a'
            )
            self.latency_label.pack(side=tk.LEFT, padx=10)
        
        if self.show_fps:
            self.fps_label = tk.Label(
                metrics_inner,
                text="📊 FPS: --",
                font=('Segoe UI', 10, 'bold'),
                fg=accent_color,
                bg='#1a1a1a'
            )
            self.fps_label.pack(side=tk.LEFT, padx=10)
        
        # Status label
        self.status_label = tk.Label(
            metrics_inner,
            text="● Running",
            font=('Segoe UI', 10, 'bold'),
            fg='#4caf50',
            bg='#1a1a1a'
        )
        self.status_label.pack(side=tk.RIGHT, padx=10)
        
        # Control buttons row
        controls_frame = tk.Frame(main_frame, bg=bg_color)
        controls_frame.pack(fill=tk.X, pady=(10, 0))
        
        # TTS Toggle button
        self.tts_button = tk.Button(
            controls_frame,
            text="🔇 TTS: OFF",
            font=('Segoe UI', 10, 'bold'),
            bg='#3a3a3a',
            fg='#ff9800',
            activebackground='#4a4a4a',
            relief=tk.FLAT,
            padx=15,
            pady=8,
            command=self._on_tts_toggle
        )
        self.tts_button.pack(side=tk.LEFT, padx=5)
        
        # Recording button
        self.record_button = tk.Button(
            controls_frame,
            text="⏺ Record",
            font=('Segoe UI', 10, 'bold'),
            bg='#3a3a3a',
            fg='#4caf50',
            activebackground='#4a4a4a',
            relief=tk.FLAT,
            padx=15,
            pady=8,
            command=self._on_record_toggle
        )
        self.record_button.pack(side=tk.LEFT, padx=5)
        
        # Custom prompt entry
        prompt_label = tk.Label(
            controls_frame,
            text="Prompt:",
            font=('Segoe UI', 10),
            fg=fg_color,
            bg=bg_color
        )
        prompt_label.pack(side=tk.LEFT, padx=(15, 5))
        
        self.prompt_entry = tk.Entry(
            controls_frame,
            font=('Segoe UI', 10),
            bg='#3a3a3a',
            fg='#ffffff',
            insertbackground='#ffffff',
            relief=tk.FLAT,
            width=30
        )
        self.prompt_entry.pack(side=tk.LEFT, padx=5)
        self.prompt_entry.bind('<Return>', lambda e: self._on_custom_prompt())
        
        # Apply prompt button
        apply_button = tk.Button(
            controls_frame,
            text="Apply",
            font=('Segoe UI', 10, 'bold'),
            bg='#00d4aa',
            fg='#000000',
            activebackground='#00ffcc',
            relief=tk.FLAT,
            padx=12,
            pady=8,
            command=self._on_custom_prompt
        )
        apply_button.pack(side=tk.LEFT, padx=5)
        
        # Help text with better styling
        help_frame = tk.Frame(main_frame, bg='#3a3a3a', relief=tk.FLAT)
        help_frame.pack(fill=tk.X, pady=(5, 0))
        
        help_text = "⌨ Q/ESC: Quit  |  P: Pause  |  S: Save  |  R: Record  |  T: Toggle TTS  |  +/-: Opacity  |  1-4: Prompts"
        help_label = tk.Label(
            help_frame,
            text=help_text,
            font=('Segoe UI', 9),
            fg='#cccccc',
            bg='#3a3a3a'
        )
        help_label.pack(pady=5)
    
    def _setup_overlay_mode(self, bg_color: str, fg_color: str, accent_color: str) -> None:
        """Set up transparent text overlay."""
        self.root.attributes('-topmost', True)
        
        # Text description
        self.text_label = tk.Label(
            self.root,
            text=self.current_text,
            font=('Segoe UI', 14, 'bold'),
            fg=fg_color,
            bg=bg_color,
            wraplength=400,
            justify='left',
            padx=20,
            pady=15
        )
        self.text_label.pack()
        
        # Metrics
        metrics_frame = tk.Frame(self.root, bg=bg_color)
        metrics_frame.pack(fill=tk.X, padx=20, pady=(0, 10))
        
        if self.show_latency:
            self.latency_label = tk.Label(
                metrics_frame,
                text="Latency: --ms",
                font=('Consolas', 9),
                fg=accent_color,
                bg=bg_color
            )
            self.latency_label.pack(side=tk.LEFT, padx=5)
        
        if self.show_fps:
            self.fps_label = tk.Label(
                metrics_frame,
                text="FPS: --",
                font=('Consolas', 9),
                fg=accent_color,
                bg=bg_color
            )
            self.fps_label.pack(side=tk.LEFT, padx=5)
    
    def _bind_keys(self) -> None:
        """Bind keyboard shortcuts."""
        # Quit
        self.root.bind('<q>', lambda e: self._on_quit())
        self.root.bind('<Q>', lambda e: self._on_quit())
        self.root.bind('<Escape>', lambda e: self._on_quit())
        
        # Pause
        self.root.bind('<p>', lambda e: self._on_pause())
        self.root.bind('<P>', lambda e: self._on_pause())
        self.root.bind('<space>', lambda e: self._on_pause())
        
        # Save
        self.root.bind('<s>', lambda e: self._on_save())
        self.root.bind('<S>', lambda e: self._on_save())
        
        # TTS toggle
        self.root.bind('<t>', lambda e: self._on_tts_toggle())
        self.root.bind('<T>', lambda e: self._on_tts_toggle())
        
        # Recording toggle
        self.root.bind('<r>', lambda e: self._on_record_toggle())
        self.root.bind('<R>', lambda e: self._on_record_toggle())
        
        # Opacity
        self.root.bind('<plus>', lambda e: self._adjust_opacity(0.05))
        self.root.bind('<equal>', lambda e: self._adjust_opacity(0.05))
        self.root.bind('<minus>', lambda e: self._adjust_opacity(-0.05))
        
        # Prompt templates (1-4)
        for i in range(1, 5):
            self.root.bind(f'<{i}>', lambda e, idx=i: self._on_prompt_change(idx))
    
    def _on_quit(self) -> None:
        """Handle quit action."""
        logger.info("Quit requested")
        if self.on_quit:
            self.on_quit()
        self.stop()
    
    def _on_pause(self) -> None:
        """Handle pause action."""
        self.paused = not self.paused
        status_text = "⏸ Paused" if self.paused else "● Running"
        status_color = '#ff9800' if self.paused else '#4caf50'
        
        if self.status_label:
            self.status_label.config(text=status_text, fg=status_color)
        
        logger.info(f"Inference {'paused' if self.paused else 'resumed'}")
        
        if self.on_pause:
            self.on_pause(self.paused)
    
    def _on_save(self) -> None:
        """Handle save action."""
        logger.info("Save requested")
        if self.on_save:
            self.on_save(self.current_text)
    
    def _on_prompt_change(self, index: int) -> None:
        """Handle prompt template change."""
        logger.info(f"Prompt template {index} requested")
        if self.on_prompt_change:
            self.on_prompt_change(index)
    
    def _on_tts_toggle(self) -> None:
        """Handle TTS toggle."""
        logger.info("TTS toggle requested")
        if self.on_tts_toggle:
            enabled = self.on_tts_toggle()
            if self.tts_button:
                if enabled:
                    self.tts_button.config(text="🔊 TTS: ON", fg='#4caf50')
                else:
                    self.tts_button.config(text="🔇 TTS: OFF", fg='#ff9800')
    
    def _on_record_toggle(self) -> None:
        """Handle recording toggle."""
        logger.info("Recording toggle requested")
        if self.on_record_toggle:
            recording = self.on_record_toggle()
            if self.record_button:
                if recording:
                    self.record_button.config(text="⏹ Stop", fg='#f44336', bg='#4a4a4a')
                else:
                    self.record_button.config(text="⏺ Record", fg='#4caf50', bg='#3a3a3a')
    
    def _on_custom_prompt(self) -> None:
        """Handle custom prompt submission."""
        if self.prompt_entry and self.on_custom_prompt:
            prompt = self.prompt_entry.get().strip()
            if prompt:
                logger.info(f"Custom prompt: {prompt}")
                self.on_custom_prompt(prompt)
    
    def set_prompt(self, prompt: str) -> None:
        """Set the prompt in the entry field.
        
        Args:
            prompt: Prompt text to display
        """
        if self.prompt_entry:
            self.prompt_entry.delete(0, tk.END)
            self.prompt_entry.insert(0, prompt)
    
    def _adjust_opacity(self, delta: float) -> None:
        """Adjust window opacity."""
        self.opacity = max(0.3, min(1.0, self.opacity + delta))
        if self.root:
            self.root.attributes('-alpha', self.opacity)
            logger.info(f"Opacity adjusted to {self.opacity:.2f}")
    
    def _on_window_close(self) -> None:
        """Handle window close event."""
        self._on_quit()
    
    def update_description(self, description: Description) -> None:
        """Update displayed description.
        
        Args:
            description: Description object with text and metadata
        """
        with self._lock:
            self.current_text = description.text
            self.current_latency = description.latency
    
    def update_frame(self, frame: np.ndarray) -> None:
        """Update camera frame display.
        
        Args:
            frame: Frame to display (BGR format)
        """
        # Don't copy - just reference (faster, less memory)
        with self._lock:
            self.current_frame = frame
    
    def update_metrics(self, metrics: MetricsCollector) -> None:
        """Update metrics display.
        
        Args:
            metrics: MetricsCollector instance
        """
        with self._lock:
            self.current_fps = metrics.avg_fps
    
    def _update_ui(self) -> None:
        """Update UI elements (called from main thread)."""
        if not self.running:
            return
        
        try:
            # Update text and metrics without lock (faster)
            if self.text_label:
                self.text_label.config(text=self.current_text)
            
            if self.latency_label and self.show_latency:
                self.latency_label.config(
                    text=f"⏱ Latency: {self.current_latency*1000:.0f}ms"
                )
            
            if self.fps_label and self.show_fps:
                self.fps_label.config(text=f"📊 FPS: {self.current_fps:.1f}")
            
            # Update frame (optimized)
            if self.canvas and self.current_frame is not None:
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                
                if canvas_width > 1 and canvas_height > 1:
                    # Fast path - use cached frame reference
                    with self._lock:
                        frame_to_display = self.current_frame
                    
                    # Convert BGR to RGB (fast)
                    frame_rgb = cv2.cvtColor(frame_to_display, cv2.COLOR_BGR2RGB)
                    
                    # Resize with fast interpolation
                    h, w = frame_rgb.shape[:2]
                    scale = min(canvas_width / w, canvas_height / h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    
                    # Use INTER_NEAREST for speed (slight quality loss but much faster)
                    resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                    
                    # Convert to PhotoImage
                    image = Image.fromarray(resized)
                    photo = ImageTk.PhotoImage(image)
                    
                    # Update canvas
                    self.canvas.delete("all")
                    x = (canvas_width - new_w) // 2
                    y = (canvas_height - new_h) // 2
                    self.canvas.create_image(x, y, anchor=tk.NW, image=photo)
                    self.canvas.image = photo  # Keep reference
            
            # Schedule next update faster
            if self.root:
                self.root.after(16, self._update_ui)  # ~60 FPS
        
        except Exception as e:
            logger.error(f"UI update error: {e}")
    
    def run(self) -> None:
        """Run the display (blocking call)."""
        if self.root is None:
            self.setup()
        
        self.running = True
        logger.info("Starting display main loop")
        
        # Start UI update loop
        self.root.after(100, self._update_ui)
        
        # Periodic check for running flag (allows Ctrl+C to work)
        def check_running():
            if not self.running:
                self.root.quit()
            else:
                self.root.after(100, check_running)
        
        self.root.after(100, check_running)
        
        # Run Tkinter main loop
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            logger.info("Display interrupted")
            self.running = False
        except Exception as e:
            logger.error(f"Display loop error: {e}")
            self.running = False
        finally:
            self.running = False
    
    def stop(self) -> None:
        """Stop the display."""
        self.running = False
        if self.root:
            try:
                # Force immediate shutdown
                self.root.quit()
                self.root.update()  # Process remaining events
                self.root.destroy()
            except Exception as e:
                logger.error(f"Error stopping display: {e}")
        logger.info("Display stopped")

