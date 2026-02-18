"""Streaming WavLM upstream expert for S3PRL evaluation.

This module provides an S3PRL-compatible upstream interface for the streaming
WavLM model, enabling evaluation on various downstream tasks like ASR, SID, etc.
"""

from __future__ import annotations

from typing import Dict, List, Union

import torch
import yaml
from focalcodec import wavlm
from torch import Tensor


class UpstreamExpert(torch.nn.Module):
    def __init__(
        self,
        ckpt,
        model_config,
        **kwargs,
    ):
        super().__init__()
        self.name = "[Streaming WavLM]"
        with open(model_config, "r") as f:
            configs = yaml.safe_load(f)
        self.model = wavlm.WavLM(causal=True, **configs)
        checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
        weights = checkpoint["student_state_dict"]
        self.model.load_state_dict(weights)

    def get_downsample_rates(self, key: str) -> int:
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
        for layer in self.model.encoder.layers:
            handles.append(layer.register_forward_hook(make_hook()))

        try:
            # Run forward pass (non-streaming for full sequence processing)
            _ = self.model(audio)
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

    def forward(self, wavs: List[Tensor]) -> Dict[str, Union[Tensor, List[Tensor]]]:
        """Process input waveforms and extract hidden states.

        Args:
            wavs: List of waveform tensors, each [T_i] (variable length)

        Returns:
            Dictionary with "hidden_states" key containing a list of tensors,
            one per transformer layer. Each tensor has shape [B, T, D].
        """
        # Pad waveforms to same length and batch
        wavs_padded = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True)
        # wavs_padded: [batch_size, max_len]

        # Extract hidden states
        hidden_states = self._extract_hidden_states(wavs_padded)

        # Skip the placeholder at index 0 for the output
        # S3PRL expects actual layer outputs
        layer_outputs = hidden_states[1:] if len(hidden_states) > 1 else hidden_states

        # Return in S3PRL format
        return {
            "hidden_states": layer_outputs,
            # Task-specific keys (all use the same hidden states)
            "PR": layer_outputs,  # Phoneme Recognition
            "ASR": layer_outputs,  # Automatic Speech Recognition
            "QbE": layer_outputs,  # Query by Example
            "SID": layer_outputs,  # Speaker Identification
            "ASV": layer_outputs,  # Automatic Speaker Verification
            "SD": layer_outputs,  # Speaker Diarization
            "ER": layer_outputs,  # Emotion Recognition
            "SF": layer_outputs,  # Slot Filling
            "SE": layer_outputs,  # Speech Enhancement
            "SS": layer_outputs,  # Source Separation
            "secret": layer_outputs,  # SUPERB hidden set
        }
