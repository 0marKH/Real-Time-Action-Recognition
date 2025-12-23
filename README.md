# Real-Time Action Recognition System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

> **Transform your camera into an AI narrator that describes what it sees in real-time**

A local, low-latency system that uses Vision-Language Models (Moondream 2 & Florence-2) to generate natural language descriptions of live camera footage. Built for accessibility, content creation, security monitoring, and research.

```
Camera Input → AI Processing → "A person waving with a smile in a bright room" 
```

**Key Highlights:**
- Real-time processing with ~300-500ms latency
- 100% local processing (no cloud, no API costs)
- Voice output for accessibility
- Records videos with auto-generated captions
- Customizable prompts on-the-fly
- Optimized for consumer GPUs (4GB+ VRAM)

---

## Demo

*Video demonstration showing the system in action*

**What it does:**
- Continuously monitors camera feed
- Generates natural language descriptions (e.g., "Person typing on keyboard", "Someone waving hello")
- Speaks descriptions aloud (optional)
- Records videos with embedded captions
- Updates descriptions as the scene changes
- Creates automatic video summaries

## Features

### Core Capabilities
- **Real-time Processing**: 2-3 descriptions per second with 300-500ms latency
- **Natural Language Output**: Complete sentences describing actions, emotions, and environment
- **Smart Change Detection**: Only processes frames when meaningful changes occur
- **Multi-threaded Architecture**: Separate threads for capture, inference, and display
- **GPU Accelerated**: Optimized for NVIDIA GPUs with efficient memory usage
- **Flexible Display**: Overlay or window mode with live camera feed
- **Local Processing**: Everything runs locally, no cloud dependencies or API costs

### Advanced Features
- **Voice Output (TTS)**: Speak descriptions aloud for accessibility applications
- **Video Recording**: Record videos with real-time caption overlay
- **Video Summaries**: Automatic timeline generation after recording stops
- **Custom Prompts**: Edit prompts on-the-fly in the UI without code changes
- **Multiple Models**: Support for Moondream 2 (detailed) and Florence-2 (fast)

## System Requirements

### Recommended
- **GPU**: NVIDIA RTX 2060 or better (8GB+ VRAM)
- **CPU**: Modern multi-core processor (i5/Ryzen 5 or better)
- **RAM**: 16GB or more
- **OS**: Windows 10/11, Linux, or macOS
- **Python**: 3.10 or higher
- **Camera**: USB webcam or built-in camera

### Minimum
- **GPU**: NVIDIA GTX 1060 (6GB VRAM) or CPU-only mode
- **CPU**: Quad-core processor
- **RAM**: 8GB
- **Python**: 3.10+

## Installation

### 1. Clone or Download

```bash
cd VLM_Rec
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install CUDA (for GPU support)

Install CUDA 11.8 or 12.1 from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads)

### 4. Install Dependencies

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
pip install -r requirements.txt
```

### 5. Verify Installation

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Quick Start

### Basic Usage

```bash
# Run with default settings
python main.py

# List available cameras
python main.py --list-cameras

# Use specific camera
python main.py --camera 1

# Adjust inference speed
python main.py --fps 3.0
```

### Display Modes

**Window Mode (default)**: Shows camera feed with text overlay
```bash
python main.py --mode window
```

**Overlay Mode**: Transparent text-only overlay (always on top)
```bash
python main.py --mode overlay
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `Q` or `ESC` | Quit application |
| `P` or `Space` | Pause/resume inference |
| `S` | Save current description to file |
| `+` / `=` | Increase window opacity |
| `-` | Decrease window opacity |
| `1-4` | Switch prompt templates |

## Configuration

### Using Config Files

Create a custom configuration:

```bash
python main.py --create-config my_config.yaml
```

Edit `my_config.yaml` and run:

```bash
python main.py --config my_config.yaml
```

### Configuration Options

See `config.yaml` for all options. Key settings:

```yaml
# Camera
camera_id: 0          # Camera device index
capture_width: 1280   # Capture resolution
capture_height: 720

# Inference
target_fps: 2.0       # Descriptions per second
quantize: true        # INT8 quantization (reduces VRAM)
compile_model: true   # torch.compile() optimization

# Change Detection
change_method: mse    # Options: mse, ssim, hash
change_threshold: 0.15  # 0-1 (lower = more sensitive)

# Display
display_mode: window  # Options: window, overlay
window_opacity: 0.85
show_latency: true
show_fps: true

# Prompt
prompt: "Describe the person's action, emotion, and environment in one sentence."
```

## Prompt Templates

Built-in templates (press 1-4 to switch):

1. **Action Focus**: "Describe what the person is doing in one sentence."
2. **Detailed**: "Describe the person's action, emotion, and environment in one sentence."
3. **Scene Overview**: "What is happening in this scene? Be concise."
4. **Emotion Focus**: "Describe the person's facial expression and emotional state."

Edit `prompts/templates.yaml` to add custom templates.

## Performance Tuning

### For Better Speed

```yaml
target_fps: 3.0
quantize: true
compile_model: true
max_tokens: 30
change_threshold: 0.20  # Less sensitive
```

### For Better Quality

```yaml
target_fps: 1.5
quantize: false
max_tokens: 75
change_threshold: 0.10  # More sensitive
```

### For Lower VRAM Usage

```yaml
quantize: true
capture_width: 640
capture_height: 480
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| `Q` or `ESC` | Quit application |
| `P` or `Space` | Pause/resume inference |
| `T` | Toggle TTS (voice output) |
| `R` | Start/stop video recording |
| `S` | Save current description |
| `+` / `=` | Increase opacity |
| `-` | Decrease opacity |
| `1-4` | Switch prompt templates |
| `Ctrl+C` | Force quit (terminal) |

## Installation Options

### Option 1: Standalone Executable (Recommended for Windows)

**Download Pre-built EXE**
- Check [Releases](../../releases) for the latest Windows executable
- No Python installation required
- Extract and run - models download automatically on first use

**Build Your Own**
```bash
python build_exe.py
```
See [EXE_BUILD_GUIDE.md](EXE_BUILD_GUIDE.md) for detailed instructions.

### Option 2: Run from Source

See the [Quick Start](#quick-start) section below for installation instructions.

## Contributing

Contributions are welcome! Please check [GITHUB_SETUP.md](GITHUB_SETUP.md) for guidelines.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Documentation

- [New Features Guide](docs/new-features-guide.md) - Complete guide to TTS, recording, and custom prompts
- [Florence-2 Guide](docs/florence-guide.md) - Alternative model for lower-end hardware
- [Hardware Requirements](docs/hardware-requirements.md) - Minimum specs and optimization
- [EXE Build Guide](EXE_BUILD_GUIDE.md) - Create standalone executable
- [GitHub Setup](GITHUB_SETUP.md) - Push to GitHub and create releases

## Use Cases

- **Accessibility**: Real-time scene descriptions for visually impaired users
- **Security Monitoring**: Natural language descriptions of surveillance footage
- **Content Creation**: Auto-generate video captions and descriptions
- **Elderly Care**: Monitor for falls and unusual activity
- **Fitness Training**: Analyze exercise form and count repetitions
- **Retail Analytics**: Track customer behavior and engagement patterns
- **Education**: Describe scientific experiments and demonstrations
- **Research**: Behavioral analysis and human-computer interaction studies

## Acknowledgments

- [Moondream 2](https://github.com/vikhyatk/moondream) - Vision-Language Model by vikhyatk
- [Florence-2](https://huggingface.co/microsoft/Florence-2-large) - Microsoft's Vision Foundation Model
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [OpenCV](https://opencv.org/) - Computer vision library

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Bug Reports**: [GitHub Issues](../../issues)
- **Feature Requests**: [GitHub Issues](../../issues)
- **Questions**: [GitHub Discussions](../../discussions)
- **Documentation**: Check the `docs/` folder

## Citation

If you use this project in your research, please cite:

```bibtex
@software{llm_rec2025,
  title={Real-Time Action Recognition System},
  author={LLM_Rec Contributors},
  year={2025},
  url={https://github.com/YOUR-USERNAME/LLM_Rec}
}
```

---

**Star this repository if you find it useful!**

## Troubleshooting

### Camera Not Detected

```bash
# List available cameras
python main.py --list-cameras

# Try different camera IDs
python main.py --camera 1
python main.py --camera 2
```

### CUDA Out of Memory

Enable quantization:
```bash
python main.py --config config.yaml  # Make sure quantize: true
```

Or reduce resolution in config.yaml:
```yaml
capture_width: 640
capture_height: 480
```

### Slow Inference

Check GPU usage:
```bash
nvidia-smi
```

Ensure you're using CUDA:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Model Download Issues

Models are downloaded automatically on first run. If download fails:
- Check internet connection
- Try again (downloads resume automatically)
- Use VPN if HuggingFace is blocked

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Camera    │────▶│   Change    │────▶│   Model     │
│  Capture    │     │  Detector   │     │ (Moondream) │
└─────────────┘     └─────────────┘     └─────────────┘
      │                                         │
      │                                         ▼
      │                                  ┌─────────────┐
      └─────────────────────────────────▶│   Display   │
                                         └─────────────┘
```

### Components

- **Camera Module**: Threaded capture with circular buffer
- **Change Detector**: MSE/SSIM/Hash-based frame comparison
- **Model Module**: Moondream 2 VLM with INT8 quantization
- **Display Module**: Tkinter-based UI with overlay support
- **Pipeline**: Orchestrates all components with queues

## Technical Details

### Model: Moondream 2

- **Architecture**: Vision-Language Model (1.6B parameters)
- **Input**: 378×378 RGB images
- **Output**: Natural language descriptions
- **Inference Time**: ~300-400ms on RTX 4060
- **VRAM Usage**: ~3.5GB (with INT8 quantization)

### Optimizations

- INT8 quantization (50% VRAM reduction)
- torch.compile() (20-30% speedup)
- Frame change detection (skips redundant processing)
- Multi-threaded pipeline (non-blocking capture)
- Circular buffer (minimizes latency)

## Expected Performance

On RTX 4060 Laptop:
- **Inference**: ~300-400ms per frame
- **End-to-end latency**: ~450-550ms
- **Throughput**: 2.5-3 descriptions/second
- **VRAM**: ~3.5GB (INT8), ~7GB (FP16)
- **CPU**: ~12-15%

## Logs and Output

Logs are saved to `logs/` directory:
- `action_recognition.log`: Detailed system logs
- `description_YYYYMMDD_HHMMSS.txt`: Saved descriptions (press 'S')

## Future Enhancements

- [ ] TTS (Text-to-Speech) output
- [ ] Action history tracking
- [ ] Multi-person tracking
- [ ] ONNX export for edge devices
- [ ] Mobile app (Android/iOS)
- [ ] Object detection overlay
- [ ] Video file processing

## Credits

- **Moondream 2**: [vikhyatk/moondream](https://github.com/vikhyatk/moondream)
- **PyTorch**: Deep learning framework
- **Transformers**: HuggingFace library
- **OpenCV**: Computer vision

## License

This project is provided as-is for educational and research purposes.

## Support

For issues and questions:
1. Check the Troubleshooting section
2. Review logs in `logs/` directory
3. Ensure all dependencies are installed correctly

---

**Note**: First run will download the Moondream 2 model (~3.5GB). This happens automatically and may take a few minutes depending on your internet connection.

