"""Utilities for biencoder sequence encoders.

Exposes a mean-pooling helper and the `BaseSeqEncoder` abstract class that implementations can extend to produce batchable sequence embeddings.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from tqdm import tqdm


def mean_pool(
    last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Mean-pool token embeddings with attention mask.

    Args:
        last_hidden_state: [B, L, H] float tensor
        attention_mask: [B, L] bool/int tensor

    Returns:
        [B, H] pooled tensor
    """
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


def _to_torch_dtype(spec: Any) -> torch.dtype:
    """Normalize dtype specifiers (torch/numpy/string) to a torch dtype."""
    if spec is None:
        return torch.float32
    if isinstance(spec, torch.dtype):
        return spec
    if isinstance(spec, str):
        cand = getattr(torch, spec, None)
        if isinstance(cand, torch.dtype):
            return cand
        if spec.startswith("torch."):
            cand = getattr(torch, spec.split(".", 1)[1], None)
            if isinstance(cand, torch.dtype):
                return cand
    try:
        np_dtype = np.dtype(spec)
    except TypeError:
        np_dtype = None
    if np_dtype is not None:
        try:
            dummy = np.empty(1, dtype=np_dtype)
            return torch.from_numpy(dummy).dtype
        except TypeError as exc:  # fall through to error below
            raise TypeError(f"Unsupported dtype specification: {spec!r}") from exc
    raise TypeError(f"Unsupported dtype specification: {spec!r}")


class BaseSeqEncoder(ABC):
    """Abstract sequence encoder that returns a single vector per sequence.

    Subclasses should implement `encode_text_batch` and return CPU tensors.
    """

    def __init__(
        self,
        device: str | None = None,
        dtype: torch.dtype = torch.float32,
        max_length: int = 1024,
        batch_size: int = 64,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = _to_torch_dtype(dtype)
        self.max_length = max_length
        self.batch_size = batch_size

    @abstractmethod
    def encode_seq_batch(self, seqs: list[str]) -> torch.Tensor:
        """Encode a batch of sequences to [B, D] tensor placed on CPU.

        Implementations may run on GPU internally but must return a CPU tensor.
        """
        raise NotImplementedError

    def encode_many(
        self,
        seqs: list[str],
        l2norm: bool = True,
        show_progress: bool = False,
        desc: str | None = None,
    ) -> np.ndarray:
        """Encode many sequences with batching, return a numpy array [N, D]."""
        iterator = range(0, len(seqs), self.batch_size)
        progress = None
        if show_progress:
            total = max(1, (len(seqs) + self.batch_size - 1) // self.batch_size)
            progress = tqdm(iterator, total=total, desc=desc or "encoding", leave=False)
            iterable = progress
        else:
            iterable = iterator

        out: list[torch.Tensor] = []
        try:
            for start in iterable:
                z = self.encode_seq_batch(seqs[start : start + self.batch_size])
                if l2norm:
                    z = torch.nn.functional.normalize(z, dim=-1)
                out.append(z)
        finally:
            if progress is not None:
                progress.close()
        return torch.cat(out, dim=0).numpy()
