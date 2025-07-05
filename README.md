# Gender Audio Classifier

A PyTorch-based audio classification system for gender detection using Wav2Vec encoder.

## Features

- **Wav2Vec models**: Transformer-based encoder models from Facebook
- **Binary Classification**: Detect gender in audio recordings

## Requirements

### Basic Requirements
- Python 3.8+
- PyTorch 1.9+
- torchaudio
- numpy
- scikit-learn
- tqdm
- tensorboard

### Wav2Vec Support
- **transformers** (for Wav2Vec support)

## Installation

```bash
git clone https://github.com/gender_classifier.git
cd gender_classifier

conda create -n classify python==3.12
conda activate classify

pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Your Data

Create label files with the following format:
```
/path/to/audio1.wav 0
/path/to/audio2.wav 1
/path/to/audio3.wav 0
```

Where:
- `0` = Female
- `1` = Male

### 2. Easy Training

```bash
bash train.sh
```

### 3. Test Your Model

```bash
bash test.sh
```

## Inference

Use the `audio_inference.py` script to run inference on new audio files using a trained model with support for Wav2Vec encoders.

### Basic Usage

Change the parameters in **infer.sh** file then run:

```bash
bash infer.sh
```

## Model Architecture

The system consists of two main components:

1. **Encoder**: Wav2Vec model for audio feature extraction
2. **Classifier**: Multi-layer neural network for binary classification

### Supported Encoders

- **Wav2Vec**: Individual file processing with HuggingFace transformers pipeline
