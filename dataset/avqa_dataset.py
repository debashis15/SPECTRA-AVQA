"""
dataset/avqa_dataset.py
-----------------------
PyTorch Dataset + collate function for Spectra-AVQA.

Assumes features have already been extracted (see `feat_script/…`) and
that the annotation JSON is a list of records of the form

    {
        "video_id":      "00000001",
        "question_id":   12345,
        "question":      "Which instrument is used first?",
        "question_content": "Which <Object> is used first?",   # template (optional)
        "type":          "Audio-Visual",                       # optional
        "templ_values":  "[\"instrument\"]",                   # optional
        "anser":         "guitar",                             # note: MUSIC-AVQA typo
        # or "answer":   "guitar"
    }

Feature files on disk:
    clip_frame_dir /<video_id>.npy     (T, 768)
    vggish_dir     /<video_id>.npy     (T, 128)
    qst_feat_dir   /<question_id>.npy  (Nq+1, 768)  where index 0 is global CLS
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


# ----------------------------------------------------------------------
def _load_answer_vocab(path: str) -> dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return {a: i for i, a in enumerate(vocab)}


def _get_answer(record: dict[str, Any]) -> str:
    # MUSIC-AVQA uses the misspelling "anser" in some versions – handle both
    return record.get("answer") or record.get("anser") or record.get("ans") or ""


# ----------------------------------------------------------------------
class AVQADataset(Dataset):
    def __init__(
        self,
        json_file: str,
        clip_frame_dir: str,
        vggish_dir: str,
        qst_feat_dir: str,
        answer_vocab_path: str,
        num_frames: int = 60,
        max_q_len: int = 77,
    ):
        super().__init__()
        with open(json_file, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

        self.clip_frame_dir = clip_frame_dir
        self.vggish_dir = vggish_dir
        self.qst_feat_dir = qst_feat_dir

        self.num_frames = num_frames
        self.max_q_len = max_q_len

        self.ans2idx = _load_answer_vocab(answer_vocab_path)
        self.idx2ans = {i: a for a, i in self.ans2idx.items()}

        # keep only samples whose answer is in the vocab (for safety)
        self.samples = [
            s for s in self.samples if _get_answer(s) in self.ans2idx
        ]
        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples found in {json_file} whose answer is in "
                f"{answer_vocab_path}"
            )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------
    def _uniform_sample(self, feat: np.ndarray, T: int) -> np.ndarray:
        """Uniformly sample / pad feat (orig_T, D) -> (T, D)."""
        if feat.ndim == 1:
            feat = feat[None, :]
        orig_T = feat.shape[0]
        if orig_T == T:
            return feat
        if orig_T == 0:
            return np.zeros((T, feat.shape[-1]), dtype=np.float32)
        if orig_T < T:
            # repeat-pad
            reps = (T + orig_T - 1) // orig_T
            feat = np.tile(feat, (reps, 1))[:T]
            return feat
        # orig_T > T : uniform indices
        idx = np.linspace(0, orig_T - 1, T).astype(np.int64)
        return feat[idx]

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.samples[idx]
        vid = str(rec["video_id"])
        qid = str(rec.get("question_id", idx))

        # -------- visual --------
        v_path = os.path.join(self.clip_frame_dir, f"{vid}.npy")
        v_feat = np.load(v_path).astype(np.float32)                # (T0, 768)
        v_feat = self._uniform_sample(v_feat, self.num_frames)      # (T,  768)

        # -------- audio --------
        a_path = os.path.join(self.vggish_dir, f"{vid}.npy")
        a_feat = np.load(a_path).astype(np.float32)                # (T0, 128)
        a_feat = self._uniform_sample(a_feat, self.num_frames)      # (T,  128)

        # -------- question --------
        q_path = os.path.join(self.qst_feat_dir, f"{qid}.npy")
        q_feat = np.load(q_path).astype(np.float32)                # (1 + Nq, 768)
        # row 0 = CLS / global, rows 1: = token-level
        q_glob = q_feat[0]                                          # (768,)
        q_tok = q_feat[1 : 1 + self.max_q_len]                      # (Nq, 768)
        # pad if shorter
        if q_tok.shape[0] < self.max_q_len:
            pad = np.zeros(
                (self.max_q_len - q_tok.shape[0], q_tok.shape[1]),
                dtype=np.float32,
            )
            q_tok = np.concatenate([q_tok, pad], axis=0)

        # -------- label --------
        label = self.ans2idx[_get_answer(rec)]

        return {
            "feat_v": torch.from_numpy(v_feat),
            "feat_a": torch.from_numpy(a_feat),
            "feat_q": torch.from_numpy(q_tok),
            "q_glob": torch.from_numpy(q_glob),
            "label":  torch.tensor(label, dtype=torch.long),
            "video_id": vid,
            "question_id": qid,
            "question": rec.get("question", ""),
            "q_type": rec.get("type", ""),
        }


# ----------------------------------------------------------------------
def avqa_collate(batch: list[dict]) -> dict[str, Any]:
    keys_tensor = ["feat_v", "feat_a", "feat_q", "q_glob", "label"]
    out: dict[str, Any] = {k: torch.stack([b[k] for b in batch], dim=0) for k in keys_tensor}
    out["video_id"]    = [b["video_id"]    for b in batch]
    out["question_id"] = [b["question_id"] for b in batch]
    out["question"]    = [b["question"]    for b in batch]
    out["q_type"]      = [b["q_type"]      for b in batch]
    return out
