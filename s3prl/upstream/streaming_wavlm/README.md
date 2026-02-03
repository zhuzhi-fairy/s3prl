# Streaming WavLM Upstream for S3PRL

This module provides S3PRL-compatible upstream models for Streaming WavLM, enabling evaluation on SUPERB benchmark downstream tasks.

## Available Models

| Model Name | Chunk Size | Latency | Description |
|------------|------------|---------|-------------|
| `streaming_wavlm` | 400ms | Medium | Default balanced configuration |
| `streaming_wavlm_200ms` | 200ms | Low | Lowest latency, may have reduced quality |
| `streaming_wavlm_400ms` | 400ms | Medium | Good balance of latency and quality |
| `streaming_wavlm_800ms` | 800ms | High | Better quality with longer context |
| `streaming_wavlm_1000ms` | 1000ms | Highest | Best quality, highest latency |
| `streaming_wavlm_offline` | 10s | N/A | Non-streaming baseline for comparison |

## Quick Start

### Prerequisites

1. Install S3PRL and dependencies:
```bash
cd Streaming_WavLM/s3prl
pip install -e .
pip install focalcodec  # Required for streaming WavLM
```

2. Prepare a checkpoint file from distillation training.

### Basic Usage

Train a downstream model with streaming WavLM features:

```bash
cd Streaming_WavLM/s3prl

# ASR task with 400ms chunks
python3 run_downstream.py -m train \
    -u streaming_wavlm_400ms \
    -k /path/to/checkpoint.pt \
    -d asr \
    -n streaming_wavlm_asr_400ms

# Speaker identification with 200ms chunks (lower latency)
python3 run_downstream.py -m train \
    -u streaming_wavlm_200ms \
    -k /path/to/checkpoint.pt \
    -d voxceleb1 \
    -n streaming_wavlm_sid_200ms

# Emotion recognition with 800ms chunks (better quality)
python3 run_downstream.py -m train \
    -u streaming_wavlm_800ms \
    -k /path/to/checkpoint.pt \
    -d emotion \
    -n streaming_wavlm_er_800ms
```

### Evaluation

Evaluate a trained downstream model:

```bash
python3 run_downstream.py -m evaluate \
    -e /path/to/downstream/experiment/dir
```

### Using with Custom Config

If you have a model configuration YAML file:

```bash
python3 run_downstream.py -m train \
    -u streaming_wavlm_400ms \
    -k /path/to/checkpoint.pt \
    -g /path/to/config.yaml \
    -d asr \
    -n experiment_name
```

## Python API

Use the upstream in your own code:

```python
import torch
import s3prl.hub as hub

# Load upstream model
model = getattr(hub, 'streaming_wavlm_400ms')(
    ckpt='/path/to/checkpoint.pt'
)
model = model.to('cuda')

# Prepare input (list of waveforms)
wavs = [torch.randn(16000).to('cuda') for _ in range(4)]  # 4 x 1 second

# Extract features
with torch.no_grad():
    output = model(wavs)
    hidden_states = output["hidden_states"]  # List of [B, T, D] tensors

# hidden_states contains one tensor per transformer layer
print(f"Number of layers: {len(hidden_states)}")
print(f"Layer 0 shape: {hidden_states[0].shape}")  # [batch, time, dim]
```

## Output Format

The `forward()` method returns a dictionary with:

- `hidden_states`: List of tensors, one per transformer layer. Each tensor has shape `[batch_size, time_steps, hidden_dim]`.
- Task-specific keys (`ASR`, `SID`, `ER`, etc.): Same as `hidden_states`, for compatibility with S3PRL downstream tasks.

## Supported Downstream Tasks

This upstream can be used with any S3PRL downstream task:

| Task | Directory | Description |
|------|-----------|-------------|
| ASR | `asr` | Automatic Speech Recognition |
| PR | `phone_linear` | Phoneme Recognition |
| SID | `voxceleb1` | Speaker Identification |
| ASV | `sv_voxceleb1` | Speaker Verification |
| SD | `diarization` | Speaker Diarization |
| ER | `emotion` | Emotion Recognition |
| IC | `fluent_commands` | Intent Classification |
| SF | `speech_commands` | Slot Filling |
| SE | `enhancement_stft` | Speech Enhancement |
| SS | `separation_stft` | Source Separation |

## Model Architecture

The streaming WavLM model uses:

- **12 transformer layers** (default)
- **1024 hidden dimension**
- **16 attention heads**
- **Causal attention** with sliding window (512 frames)
- **Lookahead size**: 3 frames (~60ms)
- **Downsampling rate**: 320x (20ms frames at 16kHz)

## Chunk Size Selection Guide

| Use Case | Recommended Chunk Size |
|----------|------------------------|
| Real-time applications | 200ms |
| Live transcription | 400ms |
| Near real-time processing | 800ms |
| Offline with streaming model | 1000ms |
| Quality comparison | offline |

## Troubleshooting

### Import Error: focalcodec not found
```bash
pip install focalcodec
```

### CUDA Out of Memory
Try reducing batch size in the downstream config or use a smaller chunk size.

### Model Loading Issues
Ensure the checkpoint contains either:
- Direct model state dict
- Dictionary with `student_state_dict` key (from distillation training)

## References

- [S3PRL Documentation](https://s3prl.github.io/s3prl/)
- [SUPERB Benchmark](https://arxiv.org/abs/2105.01051)
- [FocalCodec](https://github.com/lucadellalib/focalcodec)
- [WavLM Paper](https://arxiv.org/abs/2110.13900)
