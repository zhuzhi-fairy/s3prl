"""Reusable streaming WavLM wrapper for training and inference (portable copy).

This file is self contained so it can be dropped into other projects. It wraps
the streaming-capable `focalcodec.wavlm.WavLM` model and exposes a simple
stateful forward for chunked audio processing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import nn

try:
    from focalcodec import wavlm as focal_wavlm
except ImportError as exc:  # pragma: no cover - defensive
    raise ImportError(
        "focalcodec is required for streaming WavLM. "
        "Install with `pip install focalcodec`."
    ) from exc


@dataclass
class StreamingState:
    """Holds streaming caches for the WavLM forward."""

    curr_pos: torch.Tensor
    left_contexts: Optional[List[Optional[torch.Tensor]]]
    kv_caches: Optional[List[Optional[torch.Tensor]]]

    @classmethod
    def init(cls, model: "StreamingWavLM", batch_size: int, device: str | torch.device):
        """Create an empty state for a given batch size."""
        device = torch.device(device)
        _ = batch_size  # kept for API symmetry; not used because curr_pos is shared
        curr_pos = torch.tensor(0, dtype=torch.long, device=device)
        return cls(curr_pos=curr_pos, left_contexts=None, kv_caches=None)


class StreamingWavLM(nn.Module):
    """Lightweight wrapper around focalcodec's streaming WavLM."""

    # Known kwargs that focalcodec.wavlm.WavLM accepts
    _WAVLM_KWARGS = {
        "num_layers", "dim", "num_heads", "dropout", "causal",
        "lookahead_size", "window_size", "conv_dim", "conv_kernel",
        "conv_stride", "conv_layers", "pos_conv_kernel", "pos_conv_groups",
    }

    def __init__(
        self,
        num_layers: int = 12,
        dim: int = 1024,
        num_heads: int = 16,
        dropout: float = 0.0,
        causal: bool = True,
        lookahead_size: int = 3,
        window_size: int = 512,
        checkpoint: Optional[str] = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        """
        Args:
            num_layers: Transformer layers.
            dim: Model width.
            num_heads: Attention heads.
            dropout: Dropout rate.
            causal: If True enables streaming caches.
            lookahead_size: Lookahead frames for causal conv-pos.
            window_size: Attention window size.
            checkpoint: Optional path to a PyTorch state_dict to load.
            device, dtype: Placement for the model weights.
            **kwargs: Additional arguments (unknown kwargs are ignored).
        """
        super().__init__()
        # Filter kwargs to only include those accepted by focalcodec.wavlm.WavLM
        wavlm_kwargs = {k: v for k, v in kwargs.items() if k in self._WAVLM_KWARGS}
        self.model = focal_wavlm.WavLM(
            num_layers=num_layers,
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            causal=causal,
            lookahead_size=lookahead_size,
            window_size=window_size,
            **wavlm_kwargs,
        )
        if checkpoint:
            state = self._load_checkpoint(checkpoint)
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if missing or unexpected:
                print(f"[warn] Missing keys: {missing}, unexpected keys: {unexpected}")
        self.model.to(device=device, dtype=dtype)

    @staticmethod
    def _load_checkpoint(path: str):
        """Load checkpoint and return the student state_dict."""
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            # older torch without weights_only arg
            state = torch.load(path, map_location="cpu")
        if isinstance(state, dict):
            if "student_state_dict" in state:
                return state["student_state_dict"]
        return state

    def forward_chunk(
        self, audio_chunk: torch.Tensor, state: Optional[StreamingState] = None
    ) -> Tuple[torch.Tensor, StreamingState]:
        """
        Process a single chunk.

        Args:
            audio_chunk: Tensor [B, T] of raw 16 kHz audio.
            state: Previous StreamingState or None to reset.

        Returns:
            (features [B, T_down, dim], updated_state)
        """
        if state is None:
            state = StreamingState.init(
                self, batch_size=audio_chunk.size(0), device=audio_chunk.device
            )

        out, curr_pos, left_ctxs, kv_caches = self.model(
            audio_chunk,
            curr_pos=state.curr_pos,
            left_contexts=state.left_contexts,
            kv_caches=state.kv_caches,
            length=None,
        )
        new_state = StreamingState(
            curr_pos=curr_pos, left_contexts=left_ctxs, kv_caches=kv_caches
        )
        return out, new_state

    def stream(
        self,
        audio: torch.Tensor,
        chunk_size: int,
        state: Optional[StreamingState] = None,
    ) -> Tuple[torch.Tensor, StreamingState]:
        """
        Stream a full utterance by chunking the waveform.

        Args:
            audio: [B, T] waveform.
            chunk_size: number of samples per chunk (at 16 kHz).
            state: optional initial state (for continuing a stream).

        Returns:
            (concatenated features, final_state)
        """
        outputs: List[torch.Tensor] = []
        state = state or StreamingState.init(self, audio.size(0), audio.device)
        for start in range(0, audio.size(1), chunk_size):
            chunk = audio[:, start : start + chunk_size]
            out, state = self.forward_chunk(chunk, state)
            outputs.append(out)
        return torch.cat(outputs, dim=1), state

    def forward(self, audio: torch.Tensor, length: Optional[torch.Tensor] = None):
        """Non-streaming forward; returns only the feature tensor."""
        output, *_ = self.model(audio, length=length)
        return output


class StreamingWavLMEncoder(nn.Module):
    """Streaming WavLM followed by a projection head for downstream tasks."""

    def __init__(
        self,
        projector: nn.Module,
        wavlm_kwargs: Optional[dict] = None,
    ):
        """
        Args:
            projector: Module applied on top of frame features (e.g., pooling + linear).
            wavlm_kwargs: Passed to `StreamingWavLM`.
        """
        super().__init__()
        wavlm_kwargs = wavlm_kwargs or {}
        self.encoder = StreamingWavLM(**wavlm_kwargs)
        self.projector = projector

    def encode_stream(
        self, audio: torch.Tensor, chunk_size: int, state: Optional[StreamingState] = None
    ) -> Tuple[torch.Tensor, StreamingState]:
        feats, state = self.encoder.stream(audio, chunk_size, state)
        return self.projector(feats), state

    def forward(self, audio: torch.Tensor, length: Optional[torch.Tensor] = None):
        feats = self.encoder(audio, length=length)
        return self.projector(feats)


def mean_pool(features: torch.Tensor, length: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Simple mean pooling helper."""
    if length is None:
        return features.mean(dim=1)
    mask = torch.arange(features.size(1), device=features.device)[None, :] < length[:, None]
    masked = features * mask.unsqueeze(-1)
    return masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
