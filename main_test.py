"""
main_test.py
------------
Spectra-AVQA evaluation driver.

Usage:
    python main_test.py --config configs/config.yaml \
                        --checkpoint spectra_avqa \
                        --ckpt ./models/spectra_avqa/best.pt \
                        --gpu 0
"""

import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from dataset import AVQADataset, avqa_collate
from nets import SpectraAVQA
from utils import (
    AverageMeter, PerTypeAccuracy, accuracy, get_logger,
    load_config, save_json, set_seed,
)


# ---------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="./configs/config.yaml")
    p.add_argument("--checkpoint", default="spectra_avqa")
    p.add_argument("--ckpt", default=None,
                   help="Explicit checkpoint file. Default: "
                        "<ckpt_dir>/<checkpoint>/best.pt")
    p.add_argument("--gpu", default="0")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--save_preds", action="store_true")
    return p


# ---------------------------------------------------------------------
def main():
    args = build_parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cfg = load_config(args.config)
    if args.batch_size is not None:
        cfg["test"]["batch_size"] = args.batch_size
    set_seed(cfg["project"]["seed"])

    logger = get_logger(cfg["paths"]["log_dir"], name=f"test_{args.checkpoint}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ---------------- data ----------------
    ds_cfg = cfg["dataset"]; p = cfg["paths"]
    json_file = ds_cfg["test_json"] if args.split == "test" else ds_cfg["val_json"]
    ds = AVQADataset(
        json_file=json_file,
        clip_frame_dir=p["clip_frame_dir"],
        vggish_dir=p["vggish_dir"],
        qst_feat_dir=p["qst_feat_dir"],
        answer_vocab_path=ds_cfg["answer_vocab"],
        num_frames=ds_cfg["num_frames"],
        max_q_len=ds_cfg["max_q_len"],
    )
    loader = DataLoader(
        ds, batch_size=cfg["test"]["batch_size"], shuffle=False,
        num_workers=cfg["train"]["num_workers"], pin_memory=True,
        collate_fn=avqa_collate,
    )
    logger.info(f"split={args.split}  #samples={len(ds)}  #answers={len(ds.ans2idx)}")

    # ---------------- model ---------------
    mcfg = cfg["model"]
    model = SpectraAVQA(
        num_answers=len(ds.ans2idx),
        d_model=mcfg["d_model"],
        clip_dim=mcfg["clip_dim"],
        vggish_dim=mcfg["vggish_dim"],
        n_heads=mcfg["n_heads"],
        n_layers=mcfg["n_layers"],
        window=mcfg["sra_window"],
        n_global=mcfg["sra_global"],
        ffn_kernels=tuple(mcfg["ms_ffn_kernels"]),
        ffn_mult=mcfg["ffn_mult"],
        dropout=mcfg["dropout"],
        top_k=ds_cfg["top_k"],
    ).to(device)

    ckpt_path = args.ckpt or os.path.join(
        cfg["paths"]["ckpt_dir"], args.checkpoint, "best.pt"
    )
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck["model"])
    model.eval()
    logger.info(f"Loaded checkpoint: {ckpt_path}  (epoch {ck.get('epoch','?')})")

    # ---------------- eval ----------------
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_m = AverageMeter(); acc_m = AverageMeter()
    per_type = PerTypeAccuracy()
    preds_dump = []
    idx2ans = ds.idx2ans

    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            fv = batch["feat_v"].to(device, non_blocking=True)
            fa = batch["feat_a"].to(device, non_blocking=True)
            fq = batch["feat_q"].to(device, non_blocking=True)
            qg = batch["q_glob"].to(device, non_blocking=True)
            y  = batch["label"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=cfg["train"]["amp"]):
                logits = model(fv, fa, fq, qg)
                loss = loss_fn(logits, y)

            loss_m.update(loss.item(), y.size(0))
            acc_m.update(accuracy(logits, y), y.size(0))
            per_type.update(logits, y, batch["q_type"])

            if args.save_preds:
                pred = logits.argmax(-1).cpu().tolist()
                for i in range(y.size(0)):
                    preds_dump.append({
                        "video_id":    batch["video_id"][i],
                        "question_id": batch["question_id"][i],
                        "question":    batch["question"][i],
                        "type":        batch["q_type"][i],
                        "pred":        idx2ans.get(pred[i], "<unk>"),
                        "gt":          idx2ans.get(int(y[i].item()), "<unk>"),
                        "correct":     bool(pred[i] == int(y[i].item())),
                    })

    dur = time.time() - t0
    report = per_type.report()
    logger.info(f"[{args.split}] loss={loss_m.avg:.4f} "
                f"acc={acc_m.avg*100:.2f}%  ({dur:.1f}s, {len(ds)/max(dur,1):.1f} qps)")
    for k, v in sorted(report.items()):
        logger.info(f"   {k:20s} : {v*100:.2f}%")

    # ---------------- dump ----------------
    out_dir = os.path.join(cfg["paths"]["result_dir"], args.checkpoint)
    os.makedirs(out_dir, exist_ok=True)
    save_json(report, os.path.join(out_dir, f"{args.split}_report.json"))
    if args.save_preds:
        save_json(preds_dump, os.path.join(out_dir, f"{args.split}_preds.json"))
        logger.info(f"Saved predictions -> {out_dir}/{args.split}_preds.json")


if __name__ == "__main__":
    main()
