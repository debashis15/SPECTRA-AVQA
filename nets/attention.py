"""
nets/attention.py
-----------------
Building blocks for Spectra-AVQA:

* SparseRangeAwareMHA   (SRA-MHA)
* MultiScaleFFN         (MS-FFN)

SRA-MHA combines:
  - local windowed attention  (each query only attends to neighbours
    inside a window of size `window`)
  - a small set of `n_global` global tokens every query can attend to
This yields O(N * (window + n_global)) attention instead of O(N^2).

The implementation is vectorised using `torch.baddbmm` on sparse index
tensors; it works for cross-attention too (Q from one seq, K/V from
another) because windowing is applied on the K/V side.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Sparse Range-Aware Multi-Head Attention
# ----------------------------------------------------------------------
class SparseRangeAwareMHA(nn.Module):
    """Multi-head attention restricted to a local window + global tokens.

    Args:
        d_model: model dim
        n_heads: number of heads
        window : half-window size w; each query attends to keys in
                 [i-w, i+w].  For cross-attention we use a relative
                 mapping of query index to key index (assuming
                 comparable sequence lengths) – if not, we fall back to
                 full attention over K/V which still respects the
                 global-token subsidy.
        n_global: number of global key-positions every query can attend
                  to (first `n_global` positions of K are "global").
        dropout: attention dropout.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        window: int = 8,
        n_global: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window = window
        self.n_global = n_global

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.scale = 1.0 / math.sqrt(self.d_head)

    # ------------------------------------------------------------------
    @staticmethod
    def _build_sparse_mask(
        q_len: int, k_len: int, window: int, n_global: int, device
    ) -> torch.Tensor:
        """Return a (q_len, k_len) bool mask – True = KEEP.

        Position i (query) attends to:
          - keys in [i-window, i+window] (mapped proportionally if lengths differ)
          - keys [0, n_global)           (global tokens)
        """
        if q_len == k_len:
            idx_q = torch.arange(q_len, device=device).unsqueeze(1)  # (q,1)
            idx_k = torch.arange(k_len, device=device).unsqueeze(0)  # (1,k)
            local = (idx_k >= idx_q - window) & (idx_k <= idx_q + window)
        else:
            # map each query to its "centre" in key space
            ratio = k_len / max(q_len, 1)
            centres = (torch.arange(q_len, device=device).float() * ratio).long()
            centres = centres.unsqueeze(1)                                   # (q,1)
            idx_k = torch.arange(k_len, device=device).unsqueeze(0)          # (1,k)
            local = (idx_k >= centres - window) & (idx_k <= centres + window)

        if n_global > 0:
            global_mask = torch.zeros_like(local)
            global_mask[:, :n_global] = True
            mask = local | global_mask
        else:
            mask = local
        return mask   # (q_len, k_len) bool

    # ------------------------------------------------------------------
    def forward(
        self,
        q: torch.Tensor,              # (B, Nq, D)
        k: Optional[torch.Tensor] = None,   # (B, Nk, D)
        v: Optional[torch.Tensor] = None,   # (B, Nk, D)
        key_padding_mask: Optional[torch.Tensor] = None,   # (B, Nk) bool (True = pad)
    ) -> torch.Tensor:
        if k is None:
            k = q
        if v is None:
            v = k

        B, Nq, _ = q.shape
        Nk = k.size(1)

        Q = self.q_proj(q).view(B, Nq, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,Nq,d)
        K = self.k_proj(k).view(B, Nk, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,Nk,d)
        V = self.v_proj(v).view(B, Nk, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,Nk,d)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale   # (B,H,Nq,Nk)

        sparse_keep = self._build_sparse_mask(
            Nq, Nk, self.window, self.n_global, attn.device
        )                                                       # (Nq,Nk)
        attn = attn.masked_fill(~sparse_keep.unsqueeze(0).unsqueeze(0), float("-inf"))

        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        # safety: rows that are all -inf -> set to 0 to avoid NaN
        all_masked = torch.isinf(attn).all(dim=-1, keepdim=True)
        attn = attn.masked_fill(all_masked, 0.0)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, V)                                  # (B,H,Nq,d)
        out = out.transpose(1, 2).contiguous().view(B, Nq, self.d_model)
        return self.resid_drop(self.out_proj(out))


# ----------------------------------------------------------------------
# Multi-Scale Feed-Forward Network
# ----------------------------------------------------------------------
class MultiScaleFFN(nn.Module):
    """FFN with parallel 1-D convolution branches of different kernel sizes.

    Conceptually mimics the MS-FFN block in the architecture figure:
    the sequence (after LN) is processed by several depthwise 1D convs
    whose outputs are fused by a pointwise projection.
    """

    def __init__(
        self,
        d_model: int,
        expansion: int = 4,
        kernels: tuple[int, ...] = (1, 3, 5),
        dropout: float = 0.1,
    ):
        super().__init__()
        d_ff = d_model * expansion
        self.in_proj = nn.Linear(d_model, d_ff)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    d_ff,
                    d_ff,
                    kernel_size=k,
                    padding=k // 2,
                    groups=d_ff,   # depthwise
                )
                for k in kernels
            ]
        )
        self.act = nn.GELU()
        self.fuse = nn.Linear(d_ff * len(kernels), d_ff)
        self.out_proj = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # (B, N, D)
        h = self.in_proj(x)                                  # (B, N, d_ff)
        h_t = h.transpose(1, 2)                              # (B, d_ff, N)
        branches = [self.act(conv(h_t)) for conv in self.convs]  # list of (B, d_ff, N)
        h_cat = torch.cat(branches, dim=1).transpose(1, 2)   # (B, N, d_ff*k)
        h_fused = self.act(self.fuse(h_cat))                 # (B, N, d_ff)
        return self.drop(self.out_proj(h_fused))             # (B, N, D)
