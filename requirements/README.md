# Requirements Files

This directory contains various requirements files for different use cases.

## Installation Guide

### Basic Installation (ASR processing only)

```bash
pip install -r requirements/main.txt
pip install "nemo-toolkit[all]>=2.0.0"
```

### Complete Installation (All Features)

For the full Indic multilingual pipeline with all features:

```bash
# Core dependencies
pip install -r requirements/main.txt

# TTS and audio quality features
pip install -r requirements/tts.txt
pip install -r requirements/quality.txt

# LLM features (punctuation, advanced processing)
pip install -r requirements/llm.txt

# HuggingFace models
pip install -r requirements/huggingface.txt

# NeMo toolkit (required for ASR and LangID)
pip install "nemo-toolkit[all]>=2.0.0"
```

### Optional: Testing

```bash
pip install -r requirements/tests.txt
```

### Optional: Documentation

```bash
pip install -r requirements/docs.txt
```

---

## Individual Requirements Files

### `main.txt`
Core dependencies for basic SDP functionality:
- Audio processing (librosa, sox, ffmpeg)
- Data manipulation (pandas, numpy, dask)
- Configuration (hydra-core, omegaconf)
- Dataset loading (datasets, pyarrow)

### `tts.txt`
Dependencies for TTS-specific features:
- Audio processing (transformers, accelerate, torchaudio)
- Speaker diarization (pyannote-audio)
- Whisper models (whisperx)

### `huggingface.txt`
HuggingFace ecosystem dependencies:
- Model loading (transformers, accelerate)
- Model hub access (huggingface_hub)
- Efficient storage (safetensors)

### `llm.txt` (NEW)
LLM and advanced NLP features:
- **vLLM**: Fast LLM inference for punctuation restoration
- **SpeechBrain**: Additional language identification models
- **Faster Whisper**: Efficient ASR and language detection
- Quantization support (bitsandbytes)

### `quality.txt` (NEW)
Audio quality assessment:
- **SQUIM metrics**: PESQ, STOI, SI-SDR for TTS quality filtering
- **SNR estimation**: Signal-to-noise ratio calculation
- **Bandwidth estimation**: Frequency range analysis
- Audio analysis tools

### `tests.txt`
Testing framework and utilities:
- pytest with coverage and parallelization
- Lhotse for audio data handling in tests

### `docs.txt`
Documentation generation:
- Sphinx with modern themes
- Markdown support
- API documentation tools

### `ipl.txt`
IPL (Intelligent Processing Library) for advanced workflows

---

## GPU/CUDA Requirements

### For NVIDIA GPUs

Some features require CUDA libraries:

```bash
# For Faster Whisper
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12

# Set library path
export LD_LIBRARY_PATH=`python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))'`
```

### PyTorch Installation

Install PyTorch with CUDA support first:

```bash
# CUDA 12.1
pip install torch==2.3.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# Or CUDA 11.8
pip install torch==2.3.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
```

---

## Minimal Installation (Testing Only)

For quickly testing the codebase structure:

```bash
pip install hydra-core omegaconf pandas numpy tqdm
```

---

## Feature-Specific Installation

### For Indic Multilingual Pipeline with LLM Punctuation

```bash
pip install -r requirements/main.txt
pip install -r requirements/llm.txt
pip install -r requirements/quality.txt
pip install "nemo-toolkit[all]>=2.0.0"
```

### For TTS Training Data Preparation

```bash
pip install -r requirements/main.txt
pip install -r requirements/tts.txt
pip install -r requirements/quality.txt
pip install "nemo-toolkit[all]>=2.0.0"
```

### For ASR Inference Only

```bash
pip install -r requirements/main.txt
pip install "nemo-toolkit[asr]>=2.0.0"
```

---

## Version Policy

- **Minimum versions specified** with `>=` for compatibility
- **Upper bounds removed** where possible for flexibility
- **Exact pins** (`==`) only for packages with known compatibility issues
- **Major version constraints** (e.g., `<2.0`) only where breaking changes expected

---

## Upgrade Guide

To upgrade all packages to latest compatible versions:

```bash
pip install --upgrade -r requirements/main.txt
pip install --upgrade -r requirements/llm.txt
pip install --upgrade -r requirements/quality.txt
```

---

## Troubleshooting

### Dependency Conflicts

If you encounter conflicts:

1. Use a fresh virtual environment:
```bash
python -m venv sdp_env
source sdp_env/bin/activate  # or `sdp_env\Scripts\activate` on Windows
```

2. Install in order:
```bash
# Install PyTorch first
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Then other requirements
pip install -r requirements/main.txt
pip install -r requirements/llm.txt
```

### CUDA Issues

If CUDA libraries are not found:

```bash
# Check CUDA version
nvidia-smi

# Install matching CUDA libraries
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Docker Alternative

For a reproducible environment, use Docker:

```bash
docker build -t sdp:latest -f docker/Dockerfile .
```

---

Last updated: 2025-01-20

