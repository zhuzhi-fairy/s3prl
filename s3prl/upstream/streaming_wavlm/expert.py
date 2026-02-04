"""Streaming WavLM upstream expert for S3PRL evaluation.

This module provides an S3PRL-compatible upstream interface for the streaming
WavLM model, enabling evaluation on various downstream tasks like ASR, SID, etc.
"""

from __future__ import annotations

from typing import Dict, List, Union

import torch
import torch.nn as nn
import yaml
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from .streaming_wavlm import StreamingState, StreamingWavLM


def config_to_wavlm_kwargs(cfg: dict) -> dict:
    """Map training-style config keys (student_*) to StreamingWavLM kwargs."""
    wavlm_kwargs = cfg.get("wavlm", {}).copy()
    # Training config keys
    mapping = {
        "student_num_layers": "num_layers",
        "student_dim": "dim",
        "student_num_heads": "num_heads",
        "student_dropout": "dropout",
        "student_causal": "causal",
        "student_lookahead": "lookahead_size",
        "student_window": "window_size",
    }
    for src_k, dst_k in mapping.items():
        if src_k in cfg:
            wavlm_kwargs[dst_k] = cfg[src_k]
    return wavlm_kwargs


class UpstreamExpert(nn.Module):
    """S3PRL upstream expert for Streaming WavLM.

    This class wraps the streaming WavLM model and provides hidden state
    extraction from each transformer layer for downstream task evaluation.
    """

    def __init__(
        self,
        ckpt: str = None,
        model_config: str = None,
        chunk_size: int = None,
        **kwargs,
    ):
        """
        Args:
            ckpt: Path to the checkpoint file containing pretrained weights.
                  Can be assigned by the -k option in run_downstream.py
            model_config: Path to YAML config file for model construction.
                          Can be assigned by the -g option in run_downstream.py
            chunk_size: Chunk size in samples for streaming inference.
                        At 16kHz: 200ms=3200, 400ms=6400, 800ms=12800, 1000ms=16000
        """
        super().__init__()
        self.name = "[Streaming WavLM]"

        # S3PRL passes some kwargs that we should ignore
        s3prl_kwargs = {"refresh", "legacy"}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in s3prl_kwargs}

        # Default model configuration
        default_kwargs = {
            "num_layers": 12,
            "dim": 1024,
            "num_heads": 16,
            "dropout": 0.0,
            "causal": True,
            "lookahead_size": 3,
            "window_size": 512,
        }

        # Load config from file if provided
        if model_config is not None:
            with open(model_config, "r") as f:
                cfg = yaml.safe_load(f)
            wavlm_kwargs = config_to_wavlm_kwargs(cfg)
            default_kwargs.update(wavlm_kwargs)

        # Override with any kwargs passed directly (filtered)
        default_kwargs.update(filtered_kwargs)

        # Ensure causal mode for streaming
        default_kwargs.setdefault("causal", True)

        # Create the model
        self.model = StreamingWavLM(checkpoint=ckpt, **default_kwargs)

        # Store model dimensions for reference
        self.hidden_dim = default_kwargs.get("dim", 1024)
        self.num_layers = default_kwargs.get("num_layers", 12)

        # Chunk size for streaming inference (in samples at 16kHz)
        if chunk_size is not None:
            self.chunk_size = chunk_size
        else:
            # Default to 400ms chunks
            self.chunk_size = 6400

        print(f"{self.name} Initialized with {self.num_layers} layers, "
              f"dim={self.hidden_dim}, chunk_size={self.chunk_size} samples "
              f"({self.chunk_size / 16:.0f}ms)")

    def get_downsample_rates(self, key: str) -> int:
        """Return the downsample rate for the model.

        WavLM downsamples by 320x (20ms frames at 16kHz).
        """
        return 320

    def _extract_hidden_states(self, audio: Tensor) -> List[Tensor]:
        """Extract hidden states from each transformer layer.

        Uses forward hooks to capture intermediate layer outputs.
        Similar to student_hidden_states() in src/rd/model.py.

        Args:
            audio: Input audio tensor [B, T]

        Returns:
            List of hidden states, one per layer. Index 0 is a placeholder
            (zeros) to align with teacher indexing conventions.
        """
        hidden_states: List[Tensor] = []
        handles = []

        # Get the underlying focalcodec model
        base_model = self.model.model

        def make_hook():
            def hook(module, inp, out):
                # Output format from focalcodec: (output, next_pos, kv_cache)
                # We want just the output tensor
                if isinstance(out, tuple):
                    hidden_states.append(out[0])
                else:
                    hidden_states.append(out)
            return hook

        # Register hooks on each transformer layer
        for layer in base_model.encoder.layers:
            handles.append(layer.register_forward_hook(make_hook()))

        try:
            # Run forward pass (non-streaming for full sequence processing)
            _ = self.model.model(audio, length=None)
        finally:
            # Always remove hooks
            for h in handles:
                h.remove()

        # Add placeholder at index 0 to align with teacher indexing
        if hidden_states:
            placeholder = hidden_states[0].new_zeros(hidden_states[0].shape)
            return [placeholder] + hidden_states
        else:
            return []

    def _extract_hidden_states_streaming(self, audio: Tensor) -> List[Tensor]:
        """Extract hidden states using streaming inference.

        Processes audio in chunks and concatenates layer outputs.

        Args:
            audio: Input audio tensor [B, T]

        Returns:
            List of hidden states per layer.
        """
        batch_size = audio.size(0)
        device = audio.device

        # Storage for per-layer hidden states across chunks
        layer_outputs: List[List[Tensor]] = [[] for _ in range(self.num_layers)]

        # Initialize streaming state
        state = StreamingState.init(self.model, batch_size, device)

        # Get the underlying model
        base_model = self.model.model

        # Process audio in chunks
        for start in range(0, audio.size(1), self.chunk_size):
            chunk = audio[:, start : start + self.chunk_size]

            # Set up hooks for this chunk
            chunk_hidden: List[Tensor] = []
            handles = []

            def make_hook():
                def hook(module, inp, out):
                    if isinstance(out, tuple):
                        chunk_hidden.append(out[0])
                    else:
                        chunk_hidden.append(out)
                return hook

            for layer in base_model.encoder.layers:
                handles.append(layer.register_forward_hook(make_hook()))

            try:
                # Forward with streaming state
                out, curr_pos, left_ctxs, kv_caches = base_model(
                    chunk,
                    curr_pos=state.curr_pos,
                    left_contexts=state.left_contexts,
                    kv_caches=state.kv_caches,
                    length=None,
                )
                state = StreamingState(
                    curr_pos=curr_pos,
                    left_contexts=left_ctxs,
                    kv_caches=kv_caches,
                )
            finally:
                for h in handles:
                    h.remove()

            # Store layer outputs from this chunk
            for i, h in enumerate(chunk_hidden):
                layer_outputs[i].append(h)

        # Concatenate chunks for each layer
        hidden_states = []
        for layer_chunks in layer_outputs:
            if layer_chunks:
                hidden_states.append(torch.cat(layer_chunks, dim=1))

        # Add placeholder at index 0
        if hidden_states:
            placeholder = hidden_states[0].new_zeros(hidden_states[0].shape)
            return [placeholder] + hidden_states
        else:
            return []

    def forward(
        self, wavs: List[Tensor], streaming: bool = True
    ) -> Dict[str, Union[Tensor, List[Tensor]]]:
        """Process input waveforms and extract hidden states.

        Args:
            wavs: List of waveform tensors, each [T_i] (variable length)
            streaming: If True, use streaming inference with chunks.
                      If False, process full sequences at once.

        Returns:
            Dictionary with "hidden_states" key containing a list of tensors,
            one per transformer layer. Each tensor has shape [B, T, D].
        """
        # Pad waveforms to same length and batch
        wavs_padded = pad_sequence(wavs, batch_first=True)
        # wavs_padded: [batch_size, max_len]

        # Extract hidden states
        if streaming:
            hidden_states = self._extract_hidden_states_streaming(wavs_padded)
        else:
            hidden_states = self._extract_hidden_states(wavs_padded)

        # Skip the placeholder at index 0 for the output
        # S3PRL expects actual layer outputs
        layer_outputs = hidden_states[1:] if len(hidden_states) > 1 else hidden_states

        # Return in S3PRL format
        return {
            "hidden_states": layer_outputs,
            # Task-specific keys (all use the same hidden states)
            "PR": layer_outputs,      # Phoneme Recognition
            "ASR": layer_outputs,     # Automatic Speech Recognition
            "QbE": layer_outputs,     # Query by Example
            "SID": layer_outputs,     # Speaker Identification
            "ASV": layer_outputs,     # Automatic Speaker Verification
            "SD": layer_outputs,      # Speaker Diarization
            "ER": layer_outputs,      # Emotion Recognition
            "SF": layer_outputs,      # Slot Filling
            "SE": layer_outputs,      # Speech Enhancement
            "SS": layer_outputs,      # Source Separation
            "secret": layer_outputs,  # SUPERB hidden set
        }
