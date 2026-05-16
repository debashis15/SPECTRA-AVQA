"""
nets/ms_cmat.py
---------------
MS-CMAT – Multi-Stage Cross-Modality Attention Transformer.

Following the right-hand side of the architecture figure, three stages
progressively fuse question, visual and audio features:

  Stage 1 (Q-V):
        Q = F_q   ,  K = V = F_v
        O1' = FFN( LN( Q + SRA-MHA(Q,K,V) ) ) + residual

  Stage 2 (Q-A):
        Q = F_q   ,  K = V = F_a
        O2' = FFN( LN( Q + SRA-MHA(Q,K,V) ) ) + residual

  Stage 3 (V-guided-Q  ×  A-guided-Q):
        Q = (visual × question)  computed from Stage 1 output
        K = V = (audio × question) computed from Stage 2 output
        O3' = FFN( LN( Q + SRA-MHA(Q,K,V) ) ) + residual

Final fused feature = concat(O1', O2', O3')  (pool over tokens first).
"""

import torch
import torch.nn as nn

from .attention import SparseRangeAwareMHA


class CrossStage(nn.Module):
    """One cross-attention stage with pre-norm + FFN (Post-norm also OK)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        window: int = 8,
        n_global: int = 2,
        ffn_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_k = nn.LayerNorm(d_model)
        self.ln_v = nn.LayerNorm(d_model)
        self.sra = SparseRangeAwareMHA(
            d_model=d_model,
            n_heads=n_heads,
            window=window,
            n_global=n_global,
            dropout=dropout,
        )
        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask=None,
    ) -> torch.Tensor:
        qn = self.ln_q(q)
        kn = self.ln_k(k)
        vn = self.ln_v(v)
        attn_out = self.sra(qn, kn, vn, key_padding_mask=key_padding_mask)
        out = q + attn_out
        out = out + self.ffn(self.ln_ffn(out))
        return out


class MSCMAT(nn.Module):
    """Three-stage cross-modality fusion."""

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        window: int = 8,
        n_global: int = 2,
        ffn_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        cfg = dict(
            d_model=d_model,
            n_heads=n_heads,
            window=window,
            n_global=n_global,
            ffn_mult=ffn_mult,
            dropout=dropout,
        )
        # Stage 1: question-guided visual
        self.stage1 = CrossStage(**cfg)
        # Stage 2: question-guided audio
        self.stage2 = CrossStage(**cfg)
        # Stage 3: visual-guided-question  × audio-guided-question
        self.stage3 = CrossStage(**cfg)

        # Projections that form the question-conditioned tokens for stage 3
        self.vq_proj = nn.Linear(d_model * 2, d_model)
        self.aq_proj = nn.Linear(d_model * 2, d_model)

    # ------------------------------------------------------------------
    @staticmethod
    def _broadcast_q(q: torch.Tensor, length: int) -> torch.Tensor:
        """Broadcast / repeat question token(s) to match sequence length."""
        if q.dim() == 2:                 # (B, D)
            q = q.unsqueeze(1)           # (B, 1, D)
        if q.size(1) == length:
            return q
        if q.size(1) == 1:
            return q.expand(-1, length, -1)
        # interpolate to target length (last resort)
        return nn.functional.interpolate(
            q.transpose(1, 2), size=length, mode="linear", align_corners=False
        ).transpose(1, 2)

    # ------------------------------------------------------------------
    def forward(
        self,
        f_q: torch.Tensor,        # (B, Nq, D)  question tokens
        f_v: torch.Tensor,        # (B, Tv, D)  visual tokens (after TEMA)
        f_a: torch.Tensor,        # (B, Ta, D)  audio tokens  (after TEMA)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # --- Stage 1 : Q_question, K/V_visual ---
        # Broadcast question to visual length then cross-attend
        q_v = self._broadcast_q(f_q, f_v.size(1))
        o1 = self.stage1(q_v, f_v, f_v)                    # (B, Tv, D)

        # --- Stage 2 : Q_question, K/V_audio ---
        q_a = self._broadcast_q(f_q, f_a.size(1))
        o2 = self.stage2(q_a, f_a, f_a)                    # (B, Ta, D)

        # --- Stage 3 : (V × Q)  cross (A × Q) ---
        # Build question-conditioned visual/audio tokens by concatenating
        # the stage-1 / stage-2 outputs with the broadcast question
        # (channel concat, then project).
        vq = self.vq_proj(torch.cat([o1, q_v], dim=-1))    # (B, Tv, D)
        aq = self.aq_proj(torch.cat([o2, q_a], dim=-1))    # (B, Ta, D)
        o3 = self.stage3(vq, aq, aq)                       # (B, Tv, D)

        return o1, o2, o3
