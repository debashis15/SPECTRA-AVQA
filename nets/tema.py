"""
nets/tema.py
------------
TEMA  – Transformer-based Encoding for Modality Adaptation.

Per the diagram (left purple / pink block):

    X  --> LN --> SRA-MHA --> + X  -->  X'
    X' --> LN --> MS-FFN   --> + X' -->  X''

A TEMA encoder is a stack of N such blocks.  We use one encoder for
visual features (F_v after TempTMP) and a second one for audio
features (F_a after TempTMP).  Both run independently here and are
fused in MS-CMAT.
"""

import torch
import torch.nn as nn

from .attention import SparseRangeAwareMHA, MultiScaleFFN


class TEMABlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window: int,
        n_global: int,
        ffn_kernels: tuple[int, ...],
        ffn_mult: int,
        dropout: float,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.sra = SparseRangeAwareMHA(
            d_model=d_model,
            n_heads=n_heads,
            window=window,
            n_global=n_global,
            dropout=dropout,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.msffn = MultiScaleFFN(
            d_model=d_model,
            expansion=ffn_mult,
            kernels=tuple(ffn_kernels),
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        # Pre-norm Transformer
        h = self.ln1(x)
        x = x + self.sra(h, h, h, key_padding_mask=key_padding_mask)
        x = x + self.msffn(self.ln2(x))
        return x


class TEMAEncoder(nn.Module):
    """Stacks `n_layers` TEMA blocks + projects input to `d_model`."""

    def __init__(
        self,
        in_dim: int,
        d_model: int = 512,
        n_layers: int = 2,
        n_heads: int = 8,
        window: int = 8,
        n_global: int = 2,
        ffn_kernels=(1, 3, 5),
        ffn_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model) if in_dim != d_model else nn.Identity()
        self.layers = nn.ModuleList(
            [
                TEMABlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    window=window,
                    n_global=n_global,
                    ffn_kernels=ffn_kernels,
                    ffn_mult=ffn_mult,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.out_ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        x = self.in_proj(x)
        for blk in self.layers:
            x = blk(x, key_padding_mask=key_padding_mask)
        return self.out_ln(x)
