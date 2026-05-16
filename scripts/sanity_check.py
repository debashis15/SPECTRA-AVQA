"""
scripts/sanity_check.py
-----------------------
End-to-end smoke test for Spectra-AVQA.

Creates a tiny synthetic dataset (feature files + JSONs + answer vocab),
trains for 2 epochs and evaluates – all without any external downloads.
Useful for CI and for catching shape bugs before launching a real job.

Run:
    python scripts/sanity_check.py
"""

import json
import os
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="spectra_sanity_"))
    print(f"[sanity] tmp dir: {tmp}")

    clip_frame = tmp / "clip_frame";      clip_frame.mkdir(parents=True)
    vggish     = tmp / "vggish";          vggish.mkdir(parents=True)
    qst        = tmp / "qst";             qst.mkdir(parents=True)
    jsons      = tmp / "json";            jsons.mkdir(parents=True)
    ckpt       = tmp / "models";          ckpt.mkdir(parents=True)
    logs       = tmp / "logs";            logs.mkdir(parents=True)
    results    = tmp / "results";         results.mkdir(parents=True)

    # ---- fake features ----
    T = 20; D_CLIP = 768; D_VGG = 128; NQ = 77
    NUM_VIDS, NUM_QS_PER = 6, 4
    ANSWERS = ["guitar", "drums", "piano", "yes", "no", "two"]

    rng = np.random.default_rng(0)
    for vid in range(NUM_VIDS):
        np.save(clip_frame / f"{vid:08d}.npy",
                rng.standard_normal((T, D_CLIP)).astype(np.float32))
        np.save(vggish / f"{vid:08d}.npy",
                rng.standard_normal((T, D_VGG)).astype(np.float32))

    records = []
    qid = 1
    for vid in range(NUM_VIDS):
        for k in range(NUM_QS_PER):
            ans = ANSWERS[(vid + k) % len(ANSWERS)]
            qtype = ["Audio", "Visual", "Audio-Visual"][k % 3]
            np.save(qst / f"{qid}.npy",
                    rng.standard_normal((1 + NQ, D_CLIP)).astype(np.float32))
            records.append({
                "video_id":    f"{vid:08d}",
                "question_id": qid,
                "question":    f"Synthetic question {qid}?",
                "type":        qtype,
                "answer":      ans,
            })
            qid += 1

    # split 50/25/25
    n = len(records)
    tr, va, te = records[: n//2], records[n//2 : 3*n//4], records[3*n//4 :]
    write_json(jsons / "avqa-train.json", tr)
    write_json(jsons / "avqa-val.json",   va)
    write_json(jsons / "avqa-test.json",  te)
    write_json(jsons / "answer_vocab.json", ANSWERS)

    # ---- write config override ----
    cfg = {
        "project": {"name": "spectra-sanity", "seed": 0, "device": "cpu"},
        "paths": {
            "video_dir": str(tmp/"video"),
            "frame_dir": str(tmp/"frames"),
            "audio_dir": str(tmp/"audio"),
            "json_dir":  str(jsons),
            "clip_frame_dir": str(clip_frame),
            "clip_patch_dir": str(tmp/"patch"),
            "vggish_dir":     str(vggish),
            "qst_feat_dir":   str(qst),
            "qap_feat_dir":   str(tmp/"qap"),
            "ckpt_dir":       str(ckpt),
            "log_dir":        str(logs),
            "result_dir":     str(results),
        },
        "dataset": {
            "train_json": str(jsons / "avqa-train.json"),
            "val_json":   str(jsons / "avqa-val.json"),
            "test_json":  str(jsons / "avqa-test.json"),
            "answer_vocab": str(jsons / "answer_vocab.json"),
            "num_frames": T, "top_k": 5, "num_patches": 16, "max_q_len": NQ,
        },
        "model": {
            "d_model": 128, "n_heads": 4, "n_layers": 1, "ffn_mult": 2,
            "dropout": 0.1, "sra_window": 4, "sra_global": 1,
            "ms_ffn_kernels": [1, 3], "clip_dim": D_CLIP, "vggish_dim": D_VGG,
        },
        "train": {
            "batch_size": 2, "epochs": 2, "lr": 5e-4, "weight_decay": 1e-4,
            "warmup_epochs": 0, "num_workers": 0, "grad_clip": 1.0,
            "log_every": 1, "save_every": 1, "amp": False,
        },
        "test": {"batch_size": 1, "ckpt": str(ckpt / "san/best.pt")},
    }
    import yaml
    cfg_path = tmp / "config.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    # ---- train ----
    print("[sanity] >>> training")
    r = subprocess.run(
        [sys.executable, str(ROOT/"main_train.py"),
         "--config", str(cfg_path),
         "--checkpoint", "san",
         "--gpu", ""],
        cwd=str(ROOT),
    )
    assert r.returncode == 0, "training failed"

    # ---- test ----
    print("[sanity] >>> testing")
    r = subprocess.run(
        [sys.executable, str(ROOT/"main_test.py"),
         "--config", str(cfg_path),
         "--checkpoint", "san",
         "--split", "test",
         "--gpu", "",
         "--save_preds"],
        cwd=str(ROOT),
    )
    assert r.returncode == 0, "testing failed"

    preds_path = results / "san" / "test_preds.json"
    assert preds_path.exists(), f"predictions not saved at {preds_path}"
    print(f"[sanity] OK – predictions written to {preds_path}")

    shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
