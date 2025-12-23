"""Main entry point for Real-Time Action Recognition system."""
import argparse
import sys
from pathlib import Path
from config import Config
from src.pipeline import Pipeline
from src.camera import CameraCapture
from src.utils.logging_config import setup_logging, get_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-Time Action Recognition using Vision-Language Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python main.py
  
  # Use specific camera
  python main.py --camera 1
  
  # Load custom config
  python main.py --config my_config.yaml
  
  # List available cameras
  python main.py --list-cameras
  
  # Adjust inference speed
  python main.py --fps 3.0
  
  # Use overlay mode
  python main.py --mode overlay
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--camera',
        type=int,
        help='Camera device ID (overrides config)'
    )
    
    parser.add_argument(
        '--fps',
        type=float,
        help='Target inference FPS (overrides config)'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['window', 'overlay'],
        help='Display mode (overrides config)'
    )
    
    parser.add_argument(
        '--prompt',
        type=str,
        help='Custom prompt for descriptions (overrides config)'
    )
    
    parser.add_argument(
        '--no-quantize',
        action='store_true',
        help='Disable INT8 quantization'
    )
    
    parser.add_argument(
        '--no-compile',
        action='store_true',
        help='Disable torch.compile() optimization'
    )
    
    parser.add_argument(
        '--list-cameras',
        action='store_true',
        help='List available cameras and exit'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--create-config',
        type=str,
        metavar='FILE',
        help='Create a default config file and exit'
    )
    
    return parser.parse_args()


def list_cameras():
    """List available cameras."""
    print("Scanning for available cameras...")
    cameras = CameraCapture.list_available_cameras(max_cameras=10)
    
    if cameras:
        print(f"\nFound {len(cameras)} camera(s):")
        for i, cam_id in enumerate(cameras):
            print(f"  [{cam_id}] Camera {cam_id}")
        print("\nUse --camera <ID> to select a camera")
    else:
        print("\nNo cameras found!")
        print("Please check that:")
        print("  - A camera is connected")
        print("  - Camera drivers are installed")
        print("  - No other application is using the camera")
    
    return len(cameras) > 0


def create_default_config(filepath: str):
    """Create a default configuration file."""
    try:
        config = Config()
        config.to_file(filepath)
        print(f"Created default configuration file: {filepath}")
        print(f"\nEdit this file to customize settings, then run:")
        print(f"  python main.py --config {filepath}")
        return True
    except Exception as e:
        print(f"Error creating config file: {e}")
        return False


def main():
    """Main entry point."""
    args = parse_args()
    
    # Handle special actions
    if args.list_cameras:
        sys.exit(0 if list_cameras() else 1)
    
    if args.create_config:
        sys.exit(0 if create_default_config(args.create_config) else 1)
    
    # Load configuration
    try:
        if Path(args.config).exists():
            config = Config.from_file(args.config)
            print(f"Loaded configuration from: {args.config}")
        else:
            print(f"Config file not found: {args.config}")
            print("Using default configuration")
            config = Config()
    except Exception as e:
        print(f"Error loading config: {e}")
        print("Using default configuration")
        config = Config()
    
    # Apply command-line overrides
    if args.camera is not None:
        config.camera_id = args.camera
    
    if args.fps is not None:
        config.target_fps = args.fps
    
    if args.mode is not None:
        config.display_mode = args.mode
    
    if args.prompt is not None:
        config.prompt = args.prompt
    
    if args.no_quantize:
        config.quantize = False
    
    if args.no_compile:
        config.compile_model = False
    
    config.log_level = args.log_level
    
    # Set up logging
    logger = setup_logging(
        log_level=config.log_level,
        log_to_file=config.log_to_file,
        log_dir=config.log_dir
    )
    
    # Print banner
    print("\n" + "="*60)
    print(" Real-Time Action Recognition System")
    print(" Powered by Moondream 2 Vision-Language Model")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Camera ID: {config.camera_id}")
    print(f"  Target FPS: {config.target_fps}")
    print(f"  Display Mode: {config.display_mode}")
    print(f"  Quantization: {'Enabled' if config.quantize else 'Disabled'}")
    print(f"  Model Compile: {'Enabled' if config.compile_model else 'Disabled'}")
    print(f"  Change Detection: {config.change_method} (threshold={config.change_threshold})")
    print(f"  Prompt: {config.prompt}")
    print("\n" + "="*60 + "\n")
    
    # Create and run pipeline
    pipeline = None
    
    # Set up signal handler for Ctrl+C
    import signal
    
    def signal_handler(sig, frame):
        print("\n\n⚠️  Ctrl+C detected! Shutting down...")
        if pipeline:
            pipeline.stop()
        print("Goodbye!")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        logger.info("Starting Real-Time Action Recognition System")
        
        pipeline = Pipeline(config)
        success = pipeline.start()
        
        if not success:
            logger.error("Failed to start pipeline")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        print("\n\nShutting down...")
        if pipeline:
            pipeline.stop()
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\nFatal error: {e}")
        if pipeline:
            pipeline.stop()
        sys.exit(1)
    
    finally:
        print("\nThank you for using Real-Time Action Recognition!")
        print("Check logs/ directory for detailed logs.")
        # Small delay to ensure logs are written
        import time
        time.sleep(0.3)


if __name__ == "__main__":
    main()

