"""
nets/temp_tmp.py
----------------
Pre-Processing Module :  Temporal Top-k  +  Median Pooling.

Given per-frame features F ∈ R^{T x D} (audio or visual global token),
we:
    1.  score each frame using similarity with the question embedding
        (q-guided "TempTMP")   –  higher score = more relevant frame.
    2.  keep the Top-k most-relevant frame indices (sorted in temporal
        order to preserve monotonic time).
    3.  apply *median* pooling over a short window (size=3) for
        robustness against noisy spikes – yields F' ∈ R^{k x D}.

Output: F' ∈ R^{Tʹ x D} where Tʹ = top_k.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TempTMP(nn.Module):
    def __init__(self, top_k: int = 10, median_window: int = 3):
        super().__init__()
        assert median_window >= 1 and median_window % 2 == 1, \
            "median_window must be a positive odd integer"
        self.top_k = top_k
        self.median_window = median_window

    # ------------------------------------------------------------------
    def _score(self, f: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Cosine similarity between each frame and question vector.

        f : (B, T, D)
        q : (B, D)      -- global question embedding
        returns (B, T)
        """
        f_n = F.normalize(f, dim=-1)
        q_n = F.normalize(q, dim=-1).unsqueeze(1)      # (B,1,D)
        return (f_n * q_n).sum(-1)                      # (B,T)

    # ------------------------------------------------------------------
    def _median_pool(self, f: torch.Tensor) -> torch.Tensor:
        """Median pooling of width `median_window` over the time axis.

        f : (B, T, D)  ->  (B, T, D) with the same T (reflect padding).
        """
        if self.median_window == 1:
            return f
        w = self.median_window
        pad = w // 2
        # pad time axis with reflection
        f_p = F.pad(f.transpose(1, 2), (pad, pad), mode="reflect")   # (B, D, T+2p)
        f_p = f_p.unfold(-1, w, 1)                                   # (B, D, T, w)
        return f_p.median(dim=-1).values.transpose(1, 2)              # (B, T, D)

    # ------------------------------------------------------------------
    def forward(self, f: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """f: (B,T,D)   q: (B,D)   ->   (B, k, D)"""
        B, T, D = f.shape
        k = min(self.top_k, T)

        # 1) smooth with median pooling first for stable ranking
        f_med = self._median_pool(f)

        # 2) score frames vs. question
        scores = self._score(f_med, q)                 # (B, T)

        # 3) take top-k and re-sort by original temporal order
        topk = torch.topk(scores, k=k, dim=-1).indices     # (B, k)
        topk, _ = torch.sort(topk, dim=-1)                  # temporal order

        # 4) gather features at those indices
        idx = topk.unsqueeze(-1).expand(-1, -1, D)         # (B, k, D)
        return f_med.gather(1, idx)                        # (B, k, D)
