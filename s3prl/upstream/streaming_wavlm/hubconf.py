"""Hub configuration for Streaming WavLM upstream models.

This module registers Streaming WavLM upstream models with different chunk sizes
for S3PRL evaluation. Each entry can be used with run_downstream.py via the -u option.

Available models:
    - streaming_wavlm: Default streaming WavLM (400ms chunks)
    - streaming_wavlm_200ms: 200ms chunks (3200 samples at 16kHz)
    - streaming_wavlm_400ms: 400ms chunks (6400 samples at 16kHz)
    - streaming_wavlm_800ms: 800ms chunks (12800 samples at 16kHz)
    - streaming_wavlm_1000ms: 1000ms chunks (16000 samples at 16kHz)

Usage:
    python3 run_downstream.py -m train -u streaming_wavlm_400ms \\
        -k /path/to/checkpoint.pt -d asr -n experiment_name

    Or in Python:
        import s3prl.hub as hub
        model = getattr(hub, 'streaming_wavlm_400ms')(ckpt='/path/to/checkpoint.pt')
"""

from .expert import UpstreamExpert as _UpstreamExpert

# Chunk sizes in samples at 16kHz
CHUNK_SIZE_200MS = 3200  # 200ms * 16 samples/ms
CHUNK_SIZE_400MS = 6400  # 400ms * 16 samples/ms
CHUNK_SIZE_600MS = 9600  # 600ms * 16 samples/ms
CHUNK_SIZE_800MS = 12800  # 800ms * 16 samples/ms
CHUNK_SIZE_1000MS = 16000  # 1000ms * 16 samples/ms


def streaming_wavlm(*args, **kwargs):
    """Default Streaming WavLM upstream (400ms chunks).

    This is the default entry point for Streaming WavLM. Uses 400ms chunks
    which provides a good balance between latency and quality.

    Args:
        ckpt: Path to checkpoint file with pretrained weights (-k option)
        model_config: Path to YAML config file (-g option)
        **kwargs: Additional arguments passed to UpstreamExpert

    Returns:
        UpstreamExpert instance configured for streaming inference.

    Example:
        python3 run_downstream.py -m train -u streaming_wavlm \\
            -k checkpoint.pt -d asr -n exp1
    """
    kwargs.setdefault("chunk_size", CHUNK_SIZE_400MS)
    return _UpstreamExpert(*args, **kwargs)


def streaming_wavlm_200ms(*args, **kwargs):
    """Streaming WavLM with 200ms chunks (lowest latency).

    Uses 200ms (3200 samples) chunks for streaming inference.
    Lower latency but may have reduced quality compared to larger chunks.

    Args:
        ckpt: Path to checkpoint file with pretrained weights (-k option)
        model_config: Path to YAML config file (-g option)
        **kwargs: Additional arguments passed to UpstreamExpert

    Returns:
        UpstreamExpert instance configured for 200ms streaming.

    Example:
        python3 run_downstream.py -m train -u streaming_wavlm_200ms \\
            -k checkpoint.pt -d asr -n exp1
    """
    kwargs["chunk_size"] = CHUNK_SIZE_200MS
    return _UpstreamExpert(*args, **kwargs)


def streaming_wavlm_400ms(*args, **kwargs):
    """Streaming WavLM with 400ms chunks (balanced).

    Uses 400ms (6400 samples) chunks for streaming inference.
    Good balance between latency and quality.

    Args:
        ckpt: Path to checkpoint file with pretrained weights (-k option)
        model_config: Path to YAML config file (-g option)
        **kwargs: Additional arguments passed to UpstreamExpert

    Returns:
        UpstreamExpert instance configured for 400ms streaming.

    Example:
        python3 run_downstream.py -m train -u streaming_wavlm_400ms \\
            -k checkpoint.pt -d asr -n exp1
    """
    kwargs["chunk_size"] = CHUNK_SIZE_400MS
    return _UpstreamExpert(*args, **kwargs)


def streaming_wavlm_600ms(*args, **kwargs):
    """Streaming WavLM with 600ms chunks (balanced).

    Uses 600ms (9600 samples) chunks for streaming inference.
    Good balance between latency and quality.

    Args:
        ckpt: Path to checkpoint file with pretrained weights (-k option)
        model_config: Path to YAML config file (-g option)
        **kwargs: Additional arguments passed to UpstreamExpert

    Returns:
        UpstreamExpert instance configured for 600ms streaming.

    Example:
        python3 run_downstream.py -m train -u streaming_wavlm_600ms \\
            -k checkpoint.pt -d asr -n exp1
    """
    kwargs["chunk_size"] = CHUNK_SIZE_600MS
    return _UpstreamExpert(*args, **kwargs)


def streaming_wavlm_800ms(*args, **kwargs):
    """Streaming WavLM with 800ms chunks (higher quality).

    Uses 800ms (12800 samples) chunks for streaming inference.
    Better quality but higher latency compared to smaller chunks.

    Args:
        ckpt: Path to checkpoint file with pretrained weights (-k option)
        model_config: Path to YAML config file (-g option)
        **kwargs: Additional arguments passed to UpstreamExpert

    Returns:
        UpstreamExpert instance configured for 800ms streaming.

    Example:
        python3 run_downstream.py -m train -u streaming_wavlm_800ms \\
            -k checkpoint.pt -d asr -n exp1
    """
    kwargs["chunk_size"] = CHUNK_SIZE_800MS
    return _UpstreamExpert(*args, **kwargs)


def streaming_wavlm_1000ms(*args, **kwargs):
    """Streaming WavLM with 1000ms chunks (highest quality).

    Uses 1000ms (16000 samples) chunks for streaming inference.
    Highest quality with longer context but also highest latency.

    Args:
        ckpt: Path to checkpoint file with pretrained weights (-k option)
        model_config: Path to YAML config file (-g option)
        **kwargs: Additional arguments passed to UpstreamExpert

    Returns:
        UpstreamExpert instance configured for 1000ms streaming.

    Example:
        python3 run_downstream.py -m train -u streaming_wavlm_1000ms \\
            -k checkpoint.pt -d asr -n exp1
    """
    kwargs["chunk_size"] = CHUNK_SIZE_1000MS
    return _UpstreamExpert(*args, **kwargs)


# Non-streaming variant for comparison
def streaming_wavlm_offline(*args, **kwargs):
    """Non-streaming WavLM for comparison (processes full sequence).

    This variant processes the full audio sequence at once without chunking.
    Useful as a baseline to compare streaming vs non-streaming performance.

    Note: Still uses the causal model architecture, but processes full sequence.

    Args:
        ckpt: Path to checkpoint file with pretrained weights (-k option)
        model_config: Path to YAML config file (-g option)
        **kwargs: Additional arguments passed to UpstreamExpert

    Returns:
        UpstreamExpert instance configured for offline processing.

    Example:
        python3 run_downstream.py -m train -u streaming_wavlm_offline \\
            -k checkpoint.pt -d asr -n exp1
    """
    # Set a very large chunk size to effectively disable chunking
    kwargs["chunk_size"] = 10 * 16000  # 10 seconds
    return _UpstreamExpert(*args, **kwargs)
