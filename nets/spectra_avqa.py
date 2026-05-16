"""
nets/spectra_avqa.py
--------------------
Full Spectra-AVQA model.

Forward flow (matches the architecture diagram):

    Inputs (already extracted features):
        feat_v : (B, T, D_clip)        global CLIP frame features
        feat_a : (B, T, D_vggish)      VGGish frame features
        feat_q : (B, Nq, D_clip)       CLIP token-level question feats
        q_glob : (B, D_clip)           CLIP global question embedding

    1. Linear projections -> d_model
    2. TempTMP  (top-k, median pooling)  on V and A
    3. TEMA encoders   on V and A  (independent parameters)
    4. MS-CMAT   3-stage cross-modal fusion   -> O1, O2, O3
    5. Pool (token mean) each Oi  ->  concat  ->  classifier  -> logits
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .tema import TEMAEncoder
from .ms_cmat import MSCMAT
from .temp_tmp import TempTMP


class SpectraAVQA(nn.Module):
    def __init__(
        self,
        num_answers: int,
        d_model: int = 512,
        clip_dim: int = 768,
        vggish_dim: int = 128,
        n_heads: int = 8,
        n_layers: int = 2,
        window: int = 8,
        n_global: int = 2,
        ffn_kernels=(1, 3, 5),
        ffn_mult: int = 4,
        dropout: float = 0.1,
        top_k: int = 10,
    ):
        super().__init__()

        # ---------- TempTMP (uses question to rank frames) ----------
        self.temp_v = TempTMP(top_k=top_k)
        self.temp_a = TempTMP(top_k=top_k)

        # ---------- Project question to d_model (for temp_tmp scoring) ----------
        self.q_proj = nn.Linear(clip_dim, d_model)
        self.q_tok_proj = nn.Linear(clip_dim, d_model)
        # Project V/A to d_model (for temp_tmp scoring space)
        self.v_proj = nn.Linear(clip_dim, d_model)
        self.a_proj = nn.Linear(vggish_dim, d_model)

        # ---------- TEMA encoders ----------
        self.tema_v = TEMAEncoder(
            in_dim=d_model,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            window=window,
            n_global=n_global,
            ffn_kernels=ffn_kernels,
            ffn_mult=ffn_mult,
            dropout=dropout,
        )
        self.tema_a = TEMAEncoder(
            in_dim=d_model,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            window=window,
            n_global=n_global,
            ffn_kernels=ffn_kernels,
            ffn_mult=ffn_mult,
            dropout=dropout,
        )

        # ---------- MS-CMAT cross-modal fusion ----------
        self.mscmat = MSCMAT(
            d_model=d_model,
            n_heads=n_heads,
            window=window,
            n_global=n_global,
            ffn_mult=ffn_mult,
            dropout=dropout,
        )

        # ---------- Answer classifier ----------
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_answers),
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        feat_v: torch.Tensor,       # (B, T, 768)
        feat_a: torch.Tensor,       # (B, T, 128)
        feat_q: torch.Tensor,       # (B, Nq, 768)
        q_glob: torch.Tensor,       # (B, 768)
    ) -> torch.Tensor:

        # 1) project to d_model
        v = self.v_proj(feat_v)         # (B, T, d)
        a = self.a_proj(feat_a)         # (B, T, d)
        q_tok = self.q_tok_proj(feat_q) # (B, Nq, d)
        q_vec = self.q_proj(q_glob)     # (B, d)

        # 2) Temporal Top-k Median Pooling
        v_sel = self.temp_v(v, q_vec)   # (B, k, d)
        a_sel = self.temp_a(a, q_vec)   # (B, k, d)

        # 3) TEMA encoders
        v_enc = self.tema_v(v_sel)      # (B, k, d)
        a_enc = self.tema_a(a_sel)      # (B, k, d)

        # 4) MS-CMAT cross-modal fusion
        o1, o2, o3 = self.mscmat(q_tok, v_enc, a_enc)

        # 5) pool + classify
        pooled = torch.cat([o1.mean(1), o2.mean(1), o3.mean(1)], dim=-1)  # (B, 3d)
        return self.classifier(pooled)                                     # (B, #ans)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
