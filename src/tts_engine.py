"""Text-to-Speech engine for voice output."""
import threading
import queue
from typing import Optional
from .utils.logging_config import get_logger

logger = get_logger("tts")


class TTSEngine:
    """Text-to-Speech engine wrapper.
    
    Speaks descriptions aloud using pyttsx3 in a separate thread.
    """
    
    def __init__(self, enabled: bool = False, rate: int = 150, volume: float = 1.0):
        """Initialize TTS engine.
        
        Args:
            enabled: Whether TTS is enabled
            rate: Speech rate (words per minute)
            volume: Volume level (0.0-1.0)
        """
        self.enabled = enabled
        self.rate = rate
        self.volume = volume
        
        self.engine = None
        self.running = False
        self.text_queue = queue.Queue(maxsize=5)
        self.thread: Optional[threading.Thread] = None
        
        if self.enabled:
            self._initialize_engine()
    
    def _initialize_engine(self) -> bool:
        """Initialize the TTS engine.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', self.rate)
            self.engine.setProperty('volume', self.volume)
            logger.info("TTS engine initialized")
            return True
        except ImportError:
            logger.error("pyttsx3 not installed. Run: pip install pyttsx3")
            self.enabled = False
            return False
        except Exception as e:
            logger.error(f"Failed to initialize TTS: {e}")
            self.enabled = False
            return False
    
    def _speak_loop(self) -> None:
        """Main speaking loop running in separate thread."""
        logger.info("TTS thread started")
        
        while self.running:
            try:
                # Get text from queue (blocking with timeout)
                text = self.text_queue.get(timeout=0.5)
                
                if self.engine and self.enabled:
                    # Speak the text
                    self.engine.say(text)
                    self.engine.runAndWait()
            
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"TTS error: {e}")
        
        logger.info("TTS thread stopped")
    
    def start(self) -> None:
        """Start the TTS thread."""
        if not self.enabled:
            logger.info("TTS is disabled")
            return
        
        if not self.engine:
            if not self._initialize_engine():
                return
        
        self.running = True
        self.thread = threading.Thread(
            target=self._speak_loop,
            daemon=True,
            name="TTSThread"
        )
        self.thread.start()
        logger.info("TTS started")
    
    def stop(self) -> None:
        """Stop the TTS thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        logger.info("TTS stopped")
    
    def speak(self, text: str) -> None:
        """Queue text to be spoken.
        
        Args:
            text: Text to speak
        """
        if not self.enabled or not self.running:
            return
        
        try:
            # Add to queue (drop oldest if full)
            self.text_queue.put_nowait(text)
        except queue.Full:
            try:
                self.text_queue.get_nowait()
            except queue.Empty:
                pass
            self.text_queue.put_nowait(text)
    
    def toggle(self) -> bool:
        """Toggle TTS on/off.
        
        Returns:
            New enabled state
        """
        self.enabled = not self.enabled
        
        if self.enabled:
            if not self.engine:
                self._initialize_engine()
            if not self.running:
                self.start()
            logger.info("TTS enabled")
        else:
            logger.info("TTS disabled")
        
        return self.enabled
    
    def set_rate(self, rate: int) -> None:
        """Set speech rate.
        
        Args:
            rate: Words per minute (typical: 100-200)
        """
        self.rate = rate
        if self.engine:
            self.engine.setProperty('rate', rate)
            logger.info(f"TTS rate set to {rate}")
    
    def set_volume(self, volume: float) -> None:
        """Set volume level.
        
        Args:
            volume: Volume level (0.0-1.0)
        """
        self.volume = max(0.0, min(1.0, volume))
        if self.engine:
            self.engine.setProperty('volume', self.volume)
            logger.info(f"TTS volume set to {self.volume}")

