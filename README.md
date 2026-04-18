# Spectra-AVQA

Audio-Visual Question Answering model with **TEMA** (Transformer-based Encoding
for Modality Adaptation) and a **3-stage MS-CMAT** (Multi-Stage Cross-Modality
Attention Transformer) head. CLIP supplies visual and question features; VGGish
supplies audio features. This work borrows data / feature-extraction conventions from [MUSIC-AVQA] and [TSPM].

[MUSIC-AVQA]: https://github.com/GeWu-Lab/MUSIC-AVQA
[TSPM]:       https://github.com/GeWu-Lab/TSPM

## Project layout
```
spectra_avqa/
├── configs/config.yaml              # single source of truth
├── nets/
│   ├── attention.py                 # SRA-MHA + MS-FFN
│   ├── tema.py                      # TEMA encoder (per modality)
│   ├── ms_cmat.py                   # 3-stage cross-modality fusion
│   ├── temp_tmp.py                  # Temporal Top-k + Median Pooling
│   └── spectra_avqa.py              # end-to-end model
├── dataset/avqa_dataset.py          # AVQADataset + collate
├── feat_script/
│   ├── extract_audio_cues/extract_audio.py
│   ├── extract_visual_frames/extract_frames.py
│   ├── extract_clip_feat/extract_frames_ViT-L14.py
│   ├── extract_clip_feat/extract_qst_ViT-L14.py
│   ├── extract_vggish_feat/extract_vggish.py
│   ├── build_answer_vocab.py
│   └── verify_features.py
├── scripts/
│   ├── extract_features.sh          # runs the whole FE pipeline
│   ├── train.sh
│   ├── test.sh
│   └── sanity_check.py              # offline smoke test (no network)
├── main_train.py
├── main_test.py
├── utils.py
└── requirements.txt
```

## Model overview (matches the figure)

1. **Feature Embedding** – CLIP ViT-L/14 for frames **and** questions; VGGish for audio. All *frozen*.
2. **Pre-Processing (`TempTMP`)** – median-pool along time, then keep the top-k frames whose cosine similarity to the (global) question embedding is highest (temporal order preserved). Applied to both V and A.
3. **TEMA** – per-modality transformer `LN → SRA-MHA → + → LN → MS-FFN → +`.
   - **SRA-MHA**: sparse attention that combines a local window of size `sra_window` with `sra_global` always-visible global tokens. Works for self- *and* cross-attention.
   - **MS-FFN**: parallel depthwise Conv1d branches (kernels 1, 3, 5), fused and projected back.
4. **MS-CMAT** – three cross-attention stages (pre-norm + FFN):
   - **Stage 1**: `Q=F_q`, `K=V=F_v` → `O₁'`
   - **Stage 2**: `Q=F_q`, `K=V=F_a` → `O₂'`
   - **Stage 3**: `Q=(V×Q)` from Stage 1, `K=V=(A×Q)` from Stage 2 → `O₃'`
5. **Answering** – `concat(mean(O₁'), mean(O₂'), mean(O₃'))` → MLP classifier.

## Installation

```bash
pip install -r requirements.txt
# VGGish weights (torchvggish downloads them on first run):
pip install git+https://github.com/harritaylor/torchvggish
```

You'll also need `ffmpeg` on your PATH for frame & audio extraction.

## Data layout

```
data/
├── video/          # raw .mp4s you downloaded (MUSIC-AVQA or your own)
├── audio/          # .wav extracted from video  (step 1)
├── frames/<vid>/   # 1-fps JPGs per video       (step 2)
├── json/
│   ├── avqa-train.json    # list of records (see dataset/avqa_dataset.py)
│   ├── avqa-val.json
│   ├── avqa-test.json
│   └── answer_vocab.json  # built automatically
└── feats/
    ├── clip_visual_frame/<vid>.npy   # (T, 768)
    ├── clip_visual_patch/<vid>.npy   # (T, P, 768)  [optional]
    ├── vggish_audio/<vid>.npy        # (T, 128)
    └── clip_question/<qid>.npy       # (1+Nq, 768)
```

Each JSON record follows the MUSIC-AVQA convention:
```json
{
  "video_id":    "00000001",
  "question_id": 12345,
  "question":    "Which instrument is used first?",
  "type":        "Audio-Visual",
  "answer":      "guitar"
}
```
The loader also accepts the original MUSIC-AVQA typo `"anser"`.

## One-shot feature extraction

```bash
VIDEO_DIR=./data/video \
JSON_DIR=./data/json \
CLIP_MODEL=openai/clip-vit-large-patch14 \
bash scripts/extract_features.sh
```

This runs, in order:
1. `extract_audio.py`  – wav per video
2. `extract_frames.py` – 1-fps JPGs per video
3. `extract_frames_ViT-L14.py` – CLIP frame features
4. `extract_vggish.py` – VGGish audio features
5. `extract_qst_ViT-L14.py` – CLIP question features
6. `build_answer_vocab.py` – answer vocabulary from train JSON
7. `verify_features.py` – sanity-checks every split has all its features

## Train

```bash
bash scripts/train.sh                      # uses configs/config.yaml
# or with overrides:
bash scripts/train.sh --batch_size 32 --lr 5e-5 --epochs 40 --top_k 10
```

Checkpoints go to `./models/<checkpoint_name>/{last.pt,best.pt}`. Best is chosen by validation accuracy.

## Test

```bash
bash scripts/test.sh                       # test split, dumps preds
SPLIT=val bash scripts/test.sh             # val split
bash scripts/test.sh --ckpt ./models/my_run/best.pt
```

Outputs:
- `results/<checkpoint_name>/test_report.json` – per-type + overall accuracy
- `results/<checkpoint_name>/test_preds.json` – per-sample predictions

## Smoke test (no dataset needed)

```bash
python scripts/sanity_check.py
```
Generates synthetic `.npy` features + JSONs in a temp dir and runs `main_train.py → main_test.py` end-to-end. This is the fastest way to check your environment is set up correctly.

## Key hyper-parameters (`configs/config.yaml`)

| param | default | meaning |
|---|---|---|
| `dataset.num_frames` | 60 | frames per clip after uniform sampling |
| `dataset.top_k`      | 10 | `Tʹ` kept by TempTMP |
| `model.d_model`      | 512 | common projection dim |
| `model.sra_window`   | 8 | local window size in SRA-MHA |
| `model.sra_global`   | 2 | #global tokens in SRA-MHA |
| `model.ms_ffn_kernels` | [1,3,5] | MS-FFN Conv1d kernel sizes |
| `model.n_layers`     | 2 | TEMA depth per modality |
| `train.lr`           | 1e-4 | AdamW learning rate |
| `train.amp`          | true | fp16 mixed-precision |

## Credits

- **MUSIC-AVQA** (Li et al., CVPR 2022) – dataset & pipeline conventions.
- **TSPM** (Li et al., ACM MM 2024) – temporal selection + question-guided attention ideas.
- **CLIP** (Radford et al.) and **VGGish** (Hershey et al.) – frozen encoders.
