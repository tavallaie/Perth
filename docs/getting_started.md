# Getting Started with SoundSignature

This guide will help you get started with the SoundSignature audio watermarking library.

## Installation

### Prerequisites

Before installing SoundSignature, make sure you have the following prerequisites:

- Python 3.8 or higher
- pip package manager

For GPU acceleration (optional):
- CUDA-compatible GPU
- PyTorch with CUDA support

### Install from PyPI

```bash
pip install SoundSignature
```

### Install from Source

```bash
git clone https://github.com/resemble-ai/SoundSignature
cd SoundSignature
pip install -e .
```

## Basic Usage

Here's a simple example of how to use SoundSignature to watermark an audio file:

```python
import librosa
import soundfile as sf
from soundsignature import PerthImplicitWatermarker

# Load audio file
audio, sample_rate = librosa.load('input.wav', sr=None)

# Initialize watermarker
watermarker = PerthImplicitWatermarker()

# Apply watermark
watermarked_audio = watermarker.apply_watermark(audio, sample_rate=sample_rate)

# Save watermarked audio
sf.write('output.wav', watermarked_audio, sample_rate)
```

To extract a watermark from an audio file:

```python
import librosa
from soundsignature import PerthImplicitWatermarker

# Load audio file
audio, sample_rate = librosa.load('output.wav', sr=None)

# Initialize watermarker
watermarker = PerthImplicitWatermarker()

# Extract watermark
watermark = watermarker.get_watermark(audio, sample_rate=sample_rate)
print(f"Extracted watermark confidence: {watermark.mean():.4f}")
```

## Command Line Usage

SoundSignature also provides a command-line interface for easy usage:

```bash
# Watermark an audio file
soundsignature input.wav -o output.wav

# Extract a watermark from a file
soundsignature input.wav --extract
```

Run `soundsignature --help` for more options and information.

## Next Steps

- Check out the [examples](../examples/) directory for more complex usage examples
- See the [API Reference](./api_reference.md) for detailed information on available functions and classes
- Learn about [watermarking techniques](./watermarking_techniques.md) implemented in SoundSignature