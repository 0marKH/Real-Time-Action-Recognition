"""Main pipeline orchestrator for the action recognition system."""
import time
import queue
import threading
import cv2
from PIL import Image
from typing import Optional
from config import Config
from .camera import CameraCapture
from .model import MoondreamVLM
from .change_detector import ChangeDetector
from .display import OverlayDisplay
from .frame_updater import FrameUpdater
from .tts_engine import TTSEngine
from .video_recorder import VideoRecorder
from .utils.metrics import MetricsCollector, Description
from .utils.logging_config import get_logger

logger = get_logger("pipeline")


class Pipeline:
    """Main pipeline orchestrating all components.
    
    Coordinates the camera capture, change detection, model inference,
    and display in a multi-threaded architecture.
    """
    
    def __init__(self, config: Config):
        """Initialize pipeline with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Components
        self.camera: Optional[CameraCapture] = None
        self.model: Optional[MoondreamVLM] = None
        self.change_detector: Optional[ChangeDetector] = None
        self.display: Optional[OverlayDisplay] = None
        self.frame_updater: Optional[FrameUpdater] = None
        self.tts: Optional[TTSEngine] = None
        self.recorder: Optional[VideoRecorder] = None
        self.metrics = MetricsCollector()
        
        # Communication queues
        self.text_queue = queue.Queue(maxsize=5)
        
        # Control flags
        self.running = False
        self.paused = False
        self._inference_thread: Optional[threading.Thread] = None
        self._display_thread: Optional[threading.Thread] = None
        
        # Current prompt
        self.current_prompt = config.prompt
        
        # Load prompt templates
        self.prompt_templates = self._load_prompt_templates()
        
        logger.info("Pipeline initialized")
    
    def _load_prompt_templates(self) -> list[str]:
        """Load prompt templates from file.
        
        Returns:
            List of prompt strings
        """
        try:
            import yaml
            from pathlib import Path
            
            templates_file = Path("prompts") / "templates.yaml"
            if templates_file.exists():
                with open(templates_file, 'r') as f:
                    templates_data = yaml.safe_load(f)
                
                # Extract prompts in order
                prompts = [
                    templates_data.get('action', {}).get('prompt', ''),
                    templates_data.get('detailed', {}).get('prompt', ''),
                    templates_data.get('scene', {}).get('prompt', ''),
                    templates_data.get('emotion', {}).get('prompt', ''),
                ]
                
                return [p for p in prompts if p]
        except Exception as e:
            logger.warning(f"Failed to load prompt templates: {e}")
        
        # Default templates
        return [
            "Describe what the person is doing in one sentence.",
            "Describe the person's action, emotion, and environment in one sentence.",
            "What is happening in this scene? Be concise.",
            "Describe the person's facial expression and emotional state."
        ]
    
    def _initialize_components(self) -> bool:
        """Initialize all pipeline components.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Initialize camera
            logger.info("Initializing camera...")
            self.camera = CameraCapture(
                camera_id=self.config.camera_id,
                width=self.config.capture_width,
                height=self.config.capture_height,
                fps=self.config.capture_fps,
                buffer_size=self.config.buffer_size
            )
            
            # Initialize model
            logger.info("Initializing model (this may take a while)...")
            self.model = MoondreamVLM(
                model_name=self.config.model_name,
                model_type=self.config.model_type,
                quantize=self.config.quantize,
                compile_model=self.config.compile_model,
                max_tokens=self.config.max_tokens
            )
            
            # Initialize change detector
            logger.info("Initializing change detector...")
            self.change_detector = ChangeDetector(
                method=self.config.change_method,
                threshold=self.config.change_threshold
            )
            
            # Initialize display
            logger.info("Initializing display...")
            self.display = OverlayDisplay(
                mode=self.config.display_mode,
                opacity=self.config.window_opacity,
                show_latency=self.config.show_latency,
                show_fps=self.config.show_fps
            )
            
            # Initialize TTS
            logger.info("Initializing TTS...")
            self.tts = TTSEngine(
                enabled=self.config.tts_enabled,
                rate=self.config.tts_rate,
                volume=self.config.tts_volume
            )
            
            # Initialize video recorder
            logger.info("Initializing video recorder...")
            self.recorder = VideoRecorder(
                output_dir=self.config.recording_dir,
                fps=self.config.recording_fps
            )
            
            # Set up display callbacks
            self.display.on_quit = self.stop
            self.display.on_pause = self._on_pause
            self.display.on_save = self._on_save
            self.display.on_prompt_change = self._on_prompt_change
            self.display.on_tts_toggle = self._on_tts_toggle
            self.display.on_record_toggle = self._on_record_toggle
            self.display.on_custom_prompt = self._on_custom_prompt
            
            # Set initial prompt in UI
            self.display.set_prompt(self.current_prompt)
            
            logger.info("All components initialized successfully")
            return True
        
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            return False
    
    def _inference_loop(self) -> None:
        """Main inference loop running in separate thread."""
        logger.info("Inference thread started")
        
        last_inference_time = 0
        min_interval = 1.0 / self.config.target_fps
        frame_count = 0
        
        while self.running:
            try:
                # Check if paused
                if self.paused:
                    time.sleep(0.1)
                    continue
                
                # Rate limiting
                elapsed = time.time() - last_inference_time
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)
                    continue
                
                # Get latest frame
                frame_obj = self.camera.get_latest_frame()
                if frame_obj is None:
                    time.sleep(0.01)
                    continue
                
                frame = frame_obj.image
                frame_count += 1
                
                # Check for changes (but always process every 10 frames to ensure updates)
                has_change = self.change_detector.has_changed(frame)
                force_process = (frame_count % 10 == 0)  # Force every 10th frame
                
                if not has_change and not force_process:
                    self.metrics.record_frame(skipped=True)
                    continue
                
                # Frame changed or forced, run inference
                self.metrics.record_frame(skipped=False)
                start_time = time.time()
                
                # Prepare image for model
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Run inference
                description_text = self.model.describe(pil_image, self.current_prompt)
                
                inference_time = time.time() - start_time
                self.metrics.record_inference(inference_time)
                
                # Create description object
                description = Description(
                    text=description_text,
                    timestamp=time.time(),
                    latency=inference_time,
                    frame_number=frame_obj.frame_number
                )
                
                # Queue for display
                try:
                    self.text_queue.put_nowait(description)
                except queue.Full:
                    # Drop oldest
                    try:
                        self.text_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.text_queue.put_nowait(description)
                
                # Speak description if TTS enabled
                if self.tts:
                    self.tts.speak(description_text)
                
                # Update recorder text
                if self.recorder:
                    self.recorder.update_text(description_text)
                
                logger.info(f"New description: {description_text[:80]}... ({inference_time*1000:.0f}ms)")
                
                last_inference_time = time.time()
            
            except Exception as e:
                logger.error(f"Error in inference loop: {e}", exc_info=True)
                time.sleep(0.1)
        
        logger.info("Inference thread stopped")
    
    def _display_update_loop(self) -> None:
        """Update display with new descriptions."""
        logger.info("Display update thread started")
        
        while self.running:
            try:
                # Get description from queue (blocking with timeout)
                try:
                    description = self.text_queue.get(timeout=0.5)
                    
                    # Update display
                    if self.display:
                        self.display.update_description(description)
                        self.display.update_metrics(self.metrics)
                
                except queue.Empty:
                    # Update metrics even if no new description
                    if self.display:
                        self.display.update_metrics(self.metrics)
            
            except Exception as e:
                logger.error(f"Error in display update loop: {e}")
                time.sleep(0.1)
        
        logger.info("Display update thread stopped")
    
    def _on_pause(self, paused: bool) -> None:
        """Handle pause event from display.
        
        Args:
            paused: Whether system is paused
        """
        self.paused = paused
        logger.info(f"Inference {'paused' if paused else 'resumed'}")
    
    def _on_save(self, text: str) -> None:
        """Handle save event from display.
        
        Args:
            text: Text to save
        """
        try:
            from pathlib import Path
            import datetime
            
            # Create logs directory
            log_dir = Path(self.config.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Save to file
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = log_dir / f"description_{timestamp}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Timestamp: {datetime.datetime.now()}\n")
                f.write(f"Prompt: {self.current_prompt}\n")
                f.write(f"\nDescription:\n{text}\n")
            
            logger.info(f"Saved description to {filename}")
        
        except Exception as e:
            logger.error(f"Failed to save description: {e}")
    
    def _on_prompt_change(self, index: int) -> None:
        """Handle prompt template change.
        
        Args:
            index: Template index (1-based)
        """
        if 1 <= index <= len(self.prompt_templates):
            self.current_prompt = self.prompt_templates[index - 1]
            self.display.set_prompt(self.current_prompt)
            logger.info(f"Switched to prompt template {index}: {self.current_prompt}")
        else:
            logger.warning(f"Invalid prompt template index: {index}")
    
    def _on_tts_toggle(self) -> bool:
        """Handle TTS toggle.
        
        Returns:
            New TTS enabled state
        """
        if self.tts:
            enabled = self.tts.toggle()
            logger.info(f"TTS {'enabled' if enabled else 'disabled'}")
            return enabled
        return False
    
    def _on_record_toggle(self) -> bool:
        """Handle recording toggle.
        
        Returns:
            True if now recording, False if stopped
        """
        if not self.recorder:
            logger.warning("Recorder not initialized")
            return False
        
        if self.recorder.is_recording():
            # Stop recording
            saved_file = self.recorder.stop_recording()
            logger.info(f"Recording saved: {saved_file}")
            
            # Generate and save summary
            summary = self.recorder.generate_summary()
            summary_file = self.recorder.save_summary()
            
            # Show summary in display
            if self.display:
                self._show_recording_summary(summary, saved_file, summary_file)
            
            return False
        else:
            # Start recording
            frame_obj = self.camera.get_latest_frame()
            if frame_obj:
                h, w = frame_obj.image.shape[:2]
                success = self.recorder.start_recording(w, h)
                if success:
                    logger.info("Recording started")
                return success
            return False
    
    def _show_recording_summary(self, summary: str, video_file: Path, summary_file: Optional[Path]) -> None:
        """Show recording summary in a popup window.
        
        Args:
            summary: Summary text
            video_file: Path to video file
            summary_file: Path to summary file
        """
        import tkinter as tk
        from tkinter import scrolledtext
        
        # Create popup window
        popup = tk.Toplevel(self.display.root)
        popup.title("Recording Summary")
        popup.geometry("700x500")
        popup.configure(bg='#2b2b2b')
        
        # Title
        title_label = tk.Label(
            popup,
            text="📊 Video Recording Complete!",
            font=('Segoe UI', 16, 'bold'),
            fg='#00d4aa',
            bg='#2b2b2b'
        )
        title_label.pack(pady=15)
        
        # File info
        info_frame = tk.Frame(popup, bg='#1a1a1a')
        info_frame.pack(fill=tk.X, padx=15, pady=5)
        
        video_label = tk.Label(
            info_frame,
            text=f"📹 Video: {video_file.name}",
            font=('Segoe UI', 10),
            fg='#ffffff',
            bg='#1a1a1a',
            anchor='w'
        )
        video_label.pack(fill=tk.X, padx=10, pady=5)
        
        if summary_file:
            summary_label = tk.Label(
                info_frame,
                text=f"📝 Summary: {summary_file.name}",
                font=('Segoe UI', 10),
                fg='#ffffff',
                bg='#1a1a1a',
                anchor='w'
            )
            summary_label.pack(fill=tk.X, padx=10, pady=5)
        
        # Summary text
        text_frame = tk.Frame(popup, bg='#2b2b2b')
        text_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        text_widget = scrolledtext.ScrolledText(
            text_frame,
            font=('Consolas', 10),
            bg='#1a1a1a',
            fg='#ffffff',
            insertbackground='#ffffff',
            wrap=tk.WORD,
            relief=tk.FLAT
        )
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert('1.0', summary)
        text_widget.config(state=tk.DISABLED)
        
        # Close button
        close_button = tk.Button(
            popup,
            text="Close",
            font=('Segoe UI', 11, 'bold'),
            bg='#00d4aa',
            fg='#000000',
            activebackground='#00ffcc',
            relief=tk.FLAT,
            padx=30,
            pady=10,
            command=popup.destroy
        )
        close_button.pack(pady=15)
        
        # Center window
        popup.update_idletasks()
        x = (popup.winfo_screenwidth() // 2) - (popup.winfo_width() // 2)
        y = (popup.winfo_screenheight() // 2) - (popup.winfo_height() // 2)
        popup.geometry(f"+{x}+{y}")
        
        # Make it modal
        popup.transient(self.display.root)
        popup.grab_set()
    
    def _on_custom_prompt(self, prompt: str) -> None:
        """Handle custom prompt.
        
        Args:
            prompt: Custom prompt text
        """
        self.current_prompt = prompt
        logger.info(f"Custom prompt set: {prompt}")
    
    def start(self) -> bool:
        """Start the pipeline.
        
        Returns:
            True if started successfully, False otherwise
        """
        if self.running:
            logger.warning("Pipeline already running")
            return False
        
        logger.info("Starting pipeline...")
        
        # Initialize components
        if not self._initialize_components():
            logger.error("Failed to initialize components")
            return False
        
        # Start camera
        if not self.camera.start():
            logger.error("Failed to start camera")
            return False
        
        # Wait for camera to warm up
        logger.info("Waiting for camera to warm up...")
        time.sleep(1.0)
        
        # Set running flag
        self.running = True
        
        # Start frame updater (for smooth camera feed and recording)
        if self.config.display_mode == "window":
            self.frame_updater = FrameUpdater(self.camera, self.display, target_fps=30, recorder=self.recorder)
            self.frame_updater.start()
        
        # Start TTS
        if self.tts:
            self.tts.start()
        
        # Start inference thread
        self._inference_thread = threading.Thread(
            target=self._inference_loop,
            daemon=True,
            name="InferenceThread"
        )
        self._inference_thread.start()
        
        # Start display update thread
        self._display_thread = threading.Thread(
            target=self._display_update_loop,
            daemon=True,
            name="DisplayUpdateThread"
        )
        self._display_thread.start()
        
        # Run display (blocking - runs in main thread)
        logger.info("Pipeline started successfully")
        
        # Set up signal handler for Ctrl+C
        import signal
        
        def signal_handler(sig, frame):
            logger.info("Ctrl+C detected, shutting down...")
            self.stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        
        try:
            self.display.run()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            self.stop()
        except Exception as e:
            logger.error(f"Display error: {e}")
            self.stop()
        
        return True
    
    def stop(self) -> None:
        """Stop the pipeline."""
        if not self.running:
            return
        
        logger.info("Stopping pipeline...")
        self.running = False
        
        # Stop recording if active
        if self.recorder and self.recorder.is_recording():
            saved_file = self.recorder.stop_recording()
            logger.info(f"Recording saved on exit: {saved_file}")
        
        # Stop TTS
        if self.tts:
            self.tts.stop()
        
        # Stop frame updater
        if self.frame_updater:
            self.frame_updater.stop()
        
        # Stop camera
        if self.camera:
            self.camera.stop()
        
        # Stop display (this exits the main loop)
        if self.display:
            self.display.stop()
        
        # Give threads a moment to finish (they're daemon threads)
        time.sleep(0.2)
        
        # Quick stats (avoid hanging)
        try:
            logger.info("=== Final Statistics ===")
            logger.info(f"Total inferences: {self.metrics.total_inferences}")
            logger.info(f"Average latency: {self.metrics.avg_inference_time:.0f}ms")
            logger.info(f"Total frames: {self.metrics.total_frames}")
        except Exception as e:
            logger.error(f"Error printing stats: {e}")
        
        logger.info("Pipeline stopped")
    
    def _print_stats(self) -> None:
        """Print final statistics (removed to avoid hanging)."""
        pass

