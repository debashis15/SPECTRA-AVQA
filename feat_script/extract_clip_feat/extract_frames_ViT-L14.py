"""
feat_script/extract_clip_feat/extract_frames_ViT-L14.py
-------------------------------------------------------
Extract per-frame CLIP visual features using the frozen CLIP image
encoder.

For every video in `frame_dir/<video_id>/`:
    1. load frames in temporal order
    2. run CLIP image encoder
    3. save global feature -> clip_frame_dir/<video_id>.npy   (T, 768)
    4. save patch tokens  -> clip_patch_dir/<video_id>.npy    (T, P, 768)

Two model flavours are supported:

    * "openai/clip-vit-large-patch14"            (default, d=768)
    * "openai/clip-vit-large-patch14-336"        (higher-res, d=768)

We use HuggingFace transformers so that both global (pooled) output
and patch tokens are easy to grab in one pass.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

try:
    from transformers import CLIPImageProcessor, CLIPVisionModel
except ImportError as e:
    raise SystemExit(
        "Please `pip install transformers pillow torch` to run this script."
    ) from e


@torch.no_grad()
def extract_one_video(
    frames_dir: Path,
    model: CLIPVisionModel,
    processor: CLIPImageProcessor,
    device: torch.device,
    batch_size: int = 32,
    num_patches_keep: int = 50,
):
    """Return (global_feats, patch_feats)

    global_feats : (T, D)
    patch_feats  : (T, P, D)   P = num_patches_keep (first P tokens excl CLS)
    """
    frame_files = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
    if len(frame_files) == 0:
        return None, None

    globals_out, patches_out = [], []
    buffer = []
    for i, fp in enumerate(frame_files):
        try:
            img = Image.open(fp).convert("RGB")
        except Exception:
            continue
        buffer.append(img)
        if len(buffer) == batch_size or i == len(frame_files) - 1:
            inputs = processor(images=buffer, return_tensors="pt").to(device)
            out = model(**inputs, output_hidden_states=False)
            # last_hidden_state : (B, 1+P, D)    pooler_output : (B, D)
            last = out.last_hidden_state            # (B, 1+P, D)
            pooled = out.pooler_output              # (B, D)
            patch_tokens = last[:, 1:, :]           # drop CLS -> (B, P, D)
            P_available = patch_tokens.size(1)
            if num_patches_keep < P_available:
                # uniform-index sub-sample
                idx = torch.linspace(
                    0, P_available - 1, num_patches_keep
                ).long().to(device)
                patch_tokens = patch_tokens.index_select(1, idx)
            globals_out.append(pooled.cpu().numpy().astype(np.float32))
            patches_out.append(patch_tokens.cpu().numpy().astype(np.float32))
            buffer = []

    if not globals_out:
        return None, None
    return (
        np.concatenate(globals_out, axis=0),       # (T, D)
        np.concatenate(patches_out, axis=0),       # (T, P, D)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_dir", default="./data/frames")
    parser.add_argument("--clip_frame_dir", default="./data/feats/clip_visual_frame")
    parser.add_argument("--clip_patch_dir", default="./data/feats/clip_visual_patch")
    parser.add_argument(
        "--clip_model", default="openai/clip-vit-large-patch14",
        help="HF id; use `openai/clip-vit-large-patch14-336` for 336px",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_patches_keep", type=int, default=50)
    parser.add_argument("--save_patches", action="store_true",
                        help="Also save (T,P,D) patch features")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.clip_frame_dir, exist_ok=True)
    if args.save_patches:
        os.makedirs(args.clip_patch_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[clip-visual] loading {args.clip_model} on {device}")

    processor = CLIPImageProcessor.from_pretrained(args.clip_model)
    model = CLIPVisionModel.from_pretrained(args.clip_model).to(device).eval()

    video_dirs = sorted([p for p in Path(args.frame_dir).iterdir() if p.is_dir()])
    print(f"[clip-visual] {len(video_dirs)} video dirs")

    for vd in tqdm(video_dirs, desc="clip-visual"):
        out_g = Path(args.clip_frame_dir) / f"{vd.name}.npy"
        out_p = Path(args.clip_patch_dir) / f"{vd.name}.npy"
        if out_g.exists() and not args.overwrite:
            continue
        g, p = extract_one_video(
            vd, model, processor, device,
            batch_size=args.batch_size,
            num_patches_keep=args.num_patches_keep,
        )
        if g is None:
            print(f"[skip] no frames in {vd}")
            continue
        np.save(out_g, g)
        if args.save_patches:
            np.save(out_p, p)


if __name__ == "__main__":
    main()
