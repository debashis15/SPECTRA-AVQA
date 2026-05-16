"""
feat_script/extract_audio_cues/extract_audio.py
-----------------------------------------------
Extract .wav audio from each video in `video_dir` and save to `audio_dir`.

Requires:  moviepy, ffmpeg on the system PATH.
"""

import argparse
import os
from pathlib import Path

from moviepy.editor import VideoFileClip
from tqdm import tqdm


def extract_audio_from_video(video_path: str, audio_path: str, sr: int = 16000):
    try:
        clip = VideoFileClip(video_path)
        if clip.audio is None:
            print(f"[SKIP] no audio track: {video_path}")
            clip.close()
            return False
        clip.audio.write_audiofile(
            audio_path, fps=sr, nbytes=2, codec="pcm_s16le",
            verbose=False, logger=None,
        )
        clip.close()
        return True
    except Exception as e:
        print(f"[ERR] {video_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", default="./data/video")
    parser.add_argument("--audio_dir", default="./data/audio")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.audio_dir, exist_ok=True)
    vids = sorted(
        [p for p in Path(args.video_dir).iterdir()
         if p.suffix.lower() in (".mp4", ".mkv", ".avi", ".mov", ".webm")]
    )
    print(f"Found {len(vids)} videos in {args.video_dir}")

    ok = 0
    for vp in tqdm(vids, desc="audio"):
        ap = Path(args.audio_dir) / f"{vp.stem}.wav"
        if ap.exists() and not args.overwrite:
            ok += 1
            continue
        if extract_audio_from_video(str(vp), str(ap), sr=args.sr):
            ok += 1
    print(f"Done: {ok}/{len(vids)} audios written to {args.audio_dir}")


if __name__ == "__main__":
    main()
