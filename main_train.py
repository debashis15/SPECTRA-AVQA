"""
main_train.py
-------------
Spectra-AVQA training driver.

Usage:
    python main_train.py --config configs/config.yaml \
                         --checkpoint spectra_avqa \
                         --gpu 0
"""

import argparse
import math
import os
import time

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from dataset import AVQADataset, avqa_collate
from nets import SpectraAVQA
from utils import (
    AverageMeter, PerTypeAccuracy, accuracy, get_logger,
    load_config, save_checkpoint, save_json, set_seed,
)


# ---------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="./configs/config.yaml")
    p.add_argument("--checkpoint", default="spectra_avqa")
    p.add_argument("--gpu", default="0")
    # CLI overrides
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--resume", default=None)
    return p


def _apply_overrides(cfg, args):
    if args.batch_size is not None:  cfg["train"]["batch_size"]  = args.batch_size
    if args.epochs     is not None:  cfg["train"]["epochs"]      = args.epochs
    if args.lr         is not None:  cfg["train"]["lr"]          = args.lr
    if args.num_workers is not None: cfg["train"]["num_workers"] = args.num_workers
    if args.top_k      is not None:  cfg["dataset"]["top_k"]     = args.top_k


def warmup_cosine(step, total_steps, warmup_steps):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * p))


# ---------------------------------------------------------------------
def build_loaders(cfg):
    ds_cfg = cfg["dataset"]; p = cfg["paths"]
    common = dict(
        clip_frame_dir=p["clip_frame_dir"],
        vggish_dir=p["vggish_dir"],
        qst_feat_dir=p["qst_feat_dir"],
        answer_vocab_path=ds_cfg["answer_vocab"],
        num_frames=ds_cfg["num_frames"],
        max_q_len=ds_cfg["max_q_len"],
    )
    train_ds = AVQADataset(json_file=ds_cfg["train_json"], **common)
    val_ds   = AVQADataset(json_file=ds_cfg["val_json"],   **common)

    tcfg = cfg["train"]
    train_loader = DataLoader(
        train_ds, batch_size=tcfg["batch_size"], shuffle=True,
        num_workers=tcfg["num_workers"], pin_memory=True,
        collate_fn=avqa_collate, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(1, tcfg["batch_size"] // 2), shuffle=False,
        num_workers=tcfg["num_workers"], pin_memory=True,
        collate_fn=avqa_collate,
    )
    return train_ds, val_ds, train_loader, val_loader


def build_model(cfg, num_answers):
    mcfg = cfg["model"]
    return SpectraAVQA(
        num_answers=num_answers,
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
        top_k=cfg["dataset"]["top_k"],
    )


# ---------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, logger, amp=False):
    model.eval()
    loss_m = AverageMeter(); acc_m = AverageMeter()
    per_type = PerTypeAccuracy()
    loss_fn = torch.nn.CrossEntropyLoss()

    for batch in loader:
        fv = batch["feat_v"].to(device, non_blocking=True)
        fa = batch["feat_a"].to(device, non_blocking=True)
        fq = batch["feat_q"].to(device, non_blocking=True)
        qg = batch["q_glob"].to(device, non_blocking=True)
        y  = batch["label"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(fv, fa, fq, qg)
            loss = loss_fn(logits, y)

        loss_m.update(loss.item(), y.size(0))
        acc_m.update(accuracy(logits, y), y.size(0))
        per_type.update(logits, y, batch["q_type"])

    report = per_type.report()
    logger.info(f"[eval] loss={loss_m.avg:.4f}  acc={acc_m.avg*100:.2f}")
    for k, v in sorted(report.items()):
        logger.info(f"       {k:20s} : {v*100:.2f}%")
    return acc_m.avg, loss_m.avg, report


# ---------------------------------------------------------------------
def main():
    args = build_parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cfg = load_config(args.config)
    _apply_overrides(cfg, args)
    set_seed(cfg["project"]["seed"])

    logger = get_logger(cfg["paths"]["log_dir"], name=f"train_{args.checkpoint}")
    logger.info("Effective config:\n" + str(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ---------------- data ----------------
    train_ds, val_ds, train_loader, val_loader = build_loaders(cfg)
    num_answers = len(train_ds.ans2idx)
    logger.info(f"#answers={num_answers}  #train={len(train_ds)}  #val={len(val_ds)}")

    # ---------------- model ---------------
    model = build_model(cfg, num_answers).to(device)
    logger.info(f"Trainable params: {model.num_parameters():,}")

    # ---------------- optim ---------------
    tcfg = cfg["train"]
    optim = AdamW(
        model.parameters(),
        lr=tcfg["lr"], weight_decay=tcfg["weight_decay"],
    )
    total_steps = tcfg["epochs"] * max(1, len(train_loader))
    warmup_steps = tcfg["warmup_epochs"] * max(1, len(train_loader))
    scheduler = LambdaLR(
        optim,
        lr_lambda=lambda s: warmup_cosine(s, total_steps, warmup_steps),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=tcfg["amp"])
    loss_fn = torch.nn.CrossEntropyLoss()

    # -------------- resume ----------------
    start_epoch = 0; best_acc = 0.0
    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        optim.load_state_dict(ck["optim"])
        scheduler.load_state_dict(ck["sched"])
        start_epoch = ck["epoch"] + 1
        best_acc = ck.get("best_acc", 0.0)
        logger.info(f"Resumed from {args.resume} (epoch {start_epoch})")

    ckpt_dir = os.path.join(cfg["paths"]["ckpt_dir"], args.checkpoint)
    os.makedirs(ckpt_dir, exist_ok=True)

    # -------------- loop -------------------
    global_step = start_epoch * len(train_loader)
    for epoch in range(start_epoch, tcfg["epochs"]):
        model.train()
        loss_m = AverageMeter(); acc_m = AverageMeter()
        t0 = time.time()

        for it, batch in enumerate(train_loader):
            fv = batch["feat_v"].to(device, non_blocking=True)
            fa = batch["feat_a"].to(device, non_blocking=True)
            fq = batch["feat_q"].to(device, non_blocking=True)
            qg = batch["q_glob"].to(device, non_blocking=True)
            y  = batch["label"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=tcfg["amp"]):
                logits = model(fv, fa, fq, qg)
                loss = loss_fn(logits, y)

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            scaler.step(optim)
            scaler.update()
            scheduler.step()

            loss_m.update(loss.item(), y.size(0))
            acc_m.update(accuracy(logits, y), y.size(0))
            global_step += 1

            if (it + 1) % tcfg["log_every"] == 0:
                logger.info(
                    f"ep {epoch+1:02d}/{tcfg['epochs']} "
                    f"it {it+1:05d}/{len(train_loader)} "
                    f"lr {optim.param_groups[0]['lr']:.2e} "
                    f"loss {loss_m.avg:.4f} acc {acc_m.avg*100:.2f}"
                )

        dur = time.time() - t0
        logger.info(
            f"[epoch {epoch+1}] train loss={loss_m.avg:.4f} "
            f"acc={acc_m.avg*100:.2f}  ({dur:.1f}s)"
        )

        # ---- eval ----
        val_acc, val_loss, val_report = evaluate(
            model, val_loader, device, logger, amp=tcfg["amp"],
        )

        # ---- save ----
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "sched": scheduler.state_dict(),
            "best_acc": max(best_acc, val_acc),
            "ans2idx": train_ds.ans2idx,
            "config": cfg,
        }
        last_path = os.path.join(ckpt_dir, "last.pt")
        save_checkpoint(state, last_path)

        if val_acc > best_acc:
            best_acc = val_acc
            best_path = os.path.join(ckpt_dir, "best.pt")
            save_checkpoint(state, best_path)
            save_json(val_report, os.path.join(ckpt_dir, "best_val_report.json"))
            logger.info(f"*** new best: {best_acc*100:.2f}% -> {best_path}")

    logger.info(f"Training done. best val acc = {best_acc*100:.2f}%")


if __name__ == "__main__":
    main()
