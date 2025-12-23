"""Vision-Language Model wrapper for Moondream 2 and Florence-2."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, BitsAndBytesConfig
from PIL import Image
import time
from typing import Optional
from .utils.logging_config import get_logger

logger = get_logger("model")


class MoondreamVLM:
    """Wrapper for Vision-Language Models (Moondream 2 and Florence-2).
    
    Handles model loading, optimization, and inference for generating
    natural language descriptions from images.
    """
    
    def __init__(
        self,
        model_name: str = "vikhyatk/moondream2",
        model_type: str = "moondream",
        quantize: bool = True,
        compile_model: bool = True,
        max_tokens: int = 50,
        device: Optional[str] = None
    ):
        """Initialize the VLM.
        
        Args:
            model_name: HuggingFace model identifier
            model_type: Type of model ('moondream' or 'florence')
            quantize: Whether to use INT8 quantization
            compile_model: Whether to compile model with torch.compile()
            max_tokens: Maximum tokens to generate
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.model_type = model_type.lower()
        self.quantize = quantize
        self.compile_model = compile_model
        self.max_tokens = max_tokens
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Model type: {self.model_type}")
        
        # Model, tokenizer, and processor
        self.model = None
        self.tokenizer = None
        self.processor = None
        
        # Load model
        self._load_model()
        
        # Warm-up
        self._warmup()
    
    def _load_model(self) -> None:
        """Load the model with optimizations."""
        logger.info(f"Loading model: {self.model_name}")
        logger.info(f"Quantization: {self.quantize}, Compile: {self.compile_model}")
        
        start_time = time.time()
        
        try:
            # Load tokenizer/processor based on model type
            logger.info("Loading tokenizer/processor...")
            if self.model_type == "florence":
                self.processor = AutoProcessor.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
            else:  # moondream
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load model with appropriate settings
            if self.quantize and self.device == "cuda":
                logger.info("Loading model with INT8 quantization...")
                try:
                    # Use BitsAndBytesConfig for quantization (new API)
                    # Use float32 compute dtype to avoid Half/Float mismatch
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        bnb_8bit_compute_dtype=torch.float32,  # Use float32 for computation
                        llm_int8_enable_fp32_cpu_offload=False
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        quantization_config=quantization_config,
                        device_map="auto",
                        trust_remote_code=True,
                        attn_implementation="eager"  # Avoid flash attention issues
                    )
                except Exception as e:
                    logger.warning(f"INT8 quantization failed: {e}")
                    logger.info("Falling back to FP16 without quantization...")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True,
                        attn_implementation="eager"
                    )
            else:
                logger.info("Loading model in FP16 without quantization...")
                if self.device == "cuda":
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                        attn_implementation="eager"
                    ).to(self.device)
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        trust_remote_code=True,
                        attn_implementation="eager"
                    ).to(self.device)
            
            # Set to evaluation mode
            self.model.eval()
            
            # Compile model (PyTorch 2.0+)
            if self.compile_model and hasattr(torch, 'compile') and self.device == "cuda":
                logger.info("Compiling model with torch.compile()...")
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                except Exception as e:
                    logger.warning(f"Model compilation failed: {e}")
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f}s")
            
            # Log memory usage
            if self.device == "cuda":
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _warmup(self) -> None:
        """Run dummy inference to initialize CUDA kernels."""
        logger.info("Warming up model...")
        
        try:
            dummy_image = Image.new('RGB', (384, 384), color='black')
            _ = self.describe(dummy_image, "Describe this image.")
            logger.info("Model warm-up complete")
        except Exception as e:
            logger.warning(f"Warm-up failed: {e}")
    
    def describe(self, image: Image.Image, prompt: str) -> str:
        """Generate description for an image.
        
        Args:
            image: PIL Image to describe
            prompt: Text prompt for description
            
        Returns:
            Generated description text
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            # Ensure image is RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Model-specific inference
            if self.model_type == "florence":
                return self._florence_describe(image, prompt)
            else:  # moondream
                return self._moondream_describe(image, prompt)
        
        except Exception as e:
            logger.error(f"Inference error: {e}", exc_info=True)
            return f"[Error: {str(e)[:50]}]"
    
    def _moondream_describe(self, image: Image.Image, prompt: str) -> str:
        """Generate description using Moondream model.
        
        Args:
            image: PIL Image
            prompt: Text prompt
            
        Returns:
            Generated description
        """
        # Resize to model input size (Moondream uses 378x378)
        image = image.resize((378, 378))
        
        # Encode image and prompt
        with torch.inference_mode():
            # Moondream 2 specific encoding
            enc_image = self.model.encode_image(image)
            
            # Generate response
            answer = self.model.answer_question(
                enc_image,
                prompt,
                self.tokenizer,
                max_new_tokens=self.max_tokens,
                do_sample=False  # Greedy decoding for speed
            )
            
            return answer.strip()
    
    def _florence_describe(self, image: Image.Image, prompt: str) -> str:
        """Generate description using Florence model.
        
        Args:
            image: PIL Image
            prompt: Text prompt
            
        Returns:
            Generated description
        """
        # Florence uses detailed captioning task
        task = "<MORE_DETAILED_CAPTION>"
        
        with torch.inference_mode():
            # Process inputs
            inputs = self.processor(
                text=task,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=self.max_tokens,
                do_sample=False,
                num_beams=3
            )
            
            # Decode
            generated_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=False
            )[0]
            
            # Parse Florence response
            parsed = self.processor.post_process_generation(
                generated_text,
                task=task,
                image_size=(image.width, image.height)
            )
            
            return parsed[task].strip()
        
        return "[Florence inference incomplete]"
    
    def batch_describe(self, images: list[Image.Image], prompt: str) -> list[str]:
        """Generate descriptions for multiple images.
        
        Args:
            images: List of PIL Images
            prompt: Text prompt for descriptions
            
        Returns:
            List of generated descriptions
        """
        return [self.describe(img, prompt) for img in images]
    
    def get_memory_usage(self) -> dict:
        """Get current GPU memory usage.
        
        Returns:
            Dictionary with memory statistics
        """
        if self.device != "cuda":
            return {"device": "cpu", "allocated_gb": 0, "reserved_gb": 0}
        
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        
        return {
            "device": "cuda",
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "max_allocated_gb": max_allocated,
        }
    
    def clear_cache(self) -> None:
        """Clear GPU cache."""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")
    
    def __del__(self):
        """Cleanup when object is deleted."""
        if self.device == "cuda":
            torch.cuda.empty_cache()

