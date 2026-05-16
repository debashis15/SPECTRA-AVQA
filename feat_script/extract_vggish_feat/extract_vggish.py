"""
feat_script/extract_vggish_feat/extract_vggish.py
-------------------------------------------------
Extract VGGish audio features for each .wav in `audio_dir`.

VGGish produces a 128-d embedding per 0.96 s segment.  We use the
`torchvggish` package which bundles the AudioSet-pretrained weights.

Output:
    vggish_dir/<video_id>.npy   shape (T, 128)
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

try:
    # https://github.com/harritaylor/torchvggish
    from torchvggish import vggish, vggish_input
except ImportError as e:
    raise SystemExit(
        "Please `pip install torch resampy soundfile numpy "
        "git+https://github.com/harritaylor/torchvggish` to run this script."
    ) from e


class VGGishExtractor:
    def __init__(self, device: torch.device, postprocess: bool = True):
        self.device = device
        self.model = vggish()
        self.model.postprocess = postprocess    # True: AudioSet-standard PCA+quantise
        self.model = self.model.to(device).eval()

    @torch.no_grad()
    def __call__(self, wav_path: str) -> np.ndarray:
        # log-mel spectrogram examples : (N, 1, 96, 64)
        examples = vggish_input.wavfile_to_examples(wav_path)
        if len(examples) == 0:
            return np.zeros((0, 128), dtype=np.float32)
        examples = torch.from_numpy(examples).float().to(self.device)
        feat = self.model.forward(examples)                          # (N, 128)
        return feat.detach().cpu().numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", default="./data/audio")
    parser.add_argument("--vggish_dir", default="./data/feats/vggish_audio")
    parser.add_argument("--postprocess", action="store_true",
                        help="Apply VGGish PCA + uint8 quantisation. "
                             "Default OFF keeps raw 128-d float32 embeddings.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.vggish_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[vggish] device={device}  postprocess={args.postprocess}")

    extractor = VGGishExtractor(device, postprocess=args.postprocess)

    wavs = sorted(Path(args.audio_dir).glob("*.wav"))
    print(f"[vggish] {len(wavs)} wav files")

    for wp in tqdm(wavs, desc="vggish"):
        out = Path(args.vggish_dir) / f"{wp.stem}.npy"
        if out.exists() and not args.overwrite:
            continue
        try:
            feat = extractor(str(wp))
        except Exception as e:
            print(f"[ERR] {wp}: {e}")
            continue
        np.save(out, feat)


if __name__ == "__main__":
    main()
