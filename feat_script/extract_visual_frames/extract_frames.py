"""
feat_script/extract_visual_frames/extract_frames.py
---------------------------------------------------
Extract `fps` frames per second from every video in `video_dir` and
save them to `frame_dir/<video_id>/%06d.jpg`.

Uses the system `ffmpeg` binary for speed & reliability.
"""

import argparse
import os
import subprocess
from pathlib import Path

from tqdm import tqdm


def extract_frames(video_path: str, out_dir: str, fps: int = 1):
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_path,
        "-vf", f"fps={fps}",
        "-qscale:v", "2",
        os.path.join(out_dir, "%06d.jpg"),
    ]
    subprocess.run(cmd, check=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", default="./data/video")
    parser.add_argument("--frame_dir", default="./data/frames")
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    vids = sorted(
        [p for p in Path(args.video_dir).iterdir()
         if p.suffix.lower() in (".mp4", ".mkv", ".avi", ".mov", ".webm")]
    )
    print(f"Found {len(vids)} videos")

    for vp in tqdm(vids, desc="frames"):
        out = Path(args.frame_dir) / vp.stem
        if out.exists() and not args.overwrite and any(out.iterdir()):
            continue
        extract_frames(str(vp), str(out), fps=args.fps)


if __name__ == "__main__":
    main()
