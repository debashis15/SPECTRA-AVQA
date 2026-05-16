"""
feat_script/extract_clip_feat/extract_qst_ViT-L14.py
----------------------------------------------------
Extract CLIP text features for every question in the annotation JSONs.

For every unique `question_id` we save a single .npy file of shape
    (1 + Nq, D)
where:
    row 0            = global (pooled / CLS) question embedding
    rows 1 .. 1+Nq   = token-level embeddings (last hidden state)

This file is consumed by `dataset/avqa_dataset.py`.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

try:
    from transformers import CLIPTokenizer, CLIPTextModel
except ImportError as e:
    raise SystemExit(
        "Please `pip install transformers torch` to run this script."
    ) from e


@torch.no_grad()
def encode_questions(
    questions: list[str],
    tokenizer: CLIPTokenizer,
    model: CLIPTextModel,
    device: torch.device,
    batch_size: int = 128,
    max_length: int = 77,
):
    """Return list of np.ndarray (1+Nq, D)."""
    all_feats = []
    for i in range(0, len(questions), batch_size):
        batch = questions[i : i + batch_size]
        inputs = tokenizer(
            batch, padding="max_length", truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(device)
        out = model(**inputs)
        last = out.last_hidden_state              # (B, L, D)
        pooled = out.pooler_output                # (B, D)
        for b in range(len(batch)):
            feat = torch.cat([pooled[b : b + 1], last[b]], dim=0)   # (1+L, D)
            all_feats.append(feat.cpu().numpy().astype(np.float32))
    return all_feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_files", nargs="+", required=True,
                        help="One or more annotation JSON files")
    parser.add_argument("--qst_feat_dir", default="./data/feats/clip_question")
    parser.add_argument("--clip_model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=77)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.qst_feat_dir, exist_ok=True)

    # 1) collect all unique (qid, question) pairs
    qid2q: dict[str, str] = {}
    for jf in args.json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
        for rec in data:
            qid = str(rec.get("question_id"))
            q = rec.get("question") or rec.get("question_content") or ""
            if qid not in qid2q:
                qid2q[qid] = q
    print(f"[clip-qst] {len(qid2q)} unique questions")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CLIPTokenizer.from_pretrained(args.clip_model)
    model = CLIPTextModel.from_pretrained(args.clip_model).to(device).eval()

    # 2) filter those needing extraction
    qids, qs = [], []
    for qid, q in qid2q.items():
        out_p = Path(args.qst_feat_dir) / f"{qid}.npy"
        if out_p.exists() and not args.overwrite:
            continue
        qids.append(qid); qs.append(q)
    print(f"[clip-qst] encoding {len(qids)} questions "
          f"(skipped {len(qid2q) - len(qids)})")

    # 3) encode & save
    for start in tqdm(range(0, len(qids), args.batch_size)):
        chunk_ids = qids[start : start + args.batch_size]
        chunk_qs  = qs[start : start + args.batch_size]
        feats = encode_questions(
            chunk_qs, tokenizer, model, device,
            batch_size=len(chunk_qs), max_length=args.max_length,
        )
        for qid, feat in zip(chunk_ids, feats):
            np.save(Path(args.qst_feat_dir) / f"{qid}.npy", feat)


if __name__ == "__main__":
    main()
