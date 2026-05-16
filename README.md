<div align="center">

# 🎵 SPECTRA-AVQA

### SPECTRA-AVQA-Net: Sparse Perceptual Enhancement with Cross-Modal Transformation for Audio-Visual Question Answering

[![Paper](https://img.shields.io/badge/Paper-Under_Review_IEEE_TCSS-red)](#citation)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-MUSIC--AVQA-purple)](https://gewu-lab.github.io/MUSIC-AVQA/)

**An Audio-Visual Question Answering framework featuring TEMA (Transformer-based Encoding for Modality Adaptation) and a 3-stage MS-CMAT (Multi-Stage Cross-Modality Attention Transformer) for music performance scene understanding.**

[Overview](#-model-overview) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Benchmarks](#-benchmark-comparison) • [Citation](#-citation)

</div>

---

## 📌 Highlights

- 🎯 **Specialized for music performance scenes** — dense, continuous audio + temporally evolving visual cues
- 🧠 **TEMA encoder** with **Sparse-Region Aware Multi-Head Attention (SRA-MHA)** and **Multi-Scale FFN (MS-FFN)**
- 🔀 **3-stage MS-CMAT** for progressive cross-modal fusion (Visual↔Question, Audio↔Question, Joint)
- ⏱️ **TempTMP** module — Temporal median pooling + Top-k question-aware frame selection
- 🔬 **Frozen, off-the-shelf encoders** — CLIP ViT-L/14 (vision + text) and VGGish (audio); no costly pre-training
- 📊 Competitive with state-of-the-art on the **MUSIC-AVQA** benchmark family

---

## 📖 Table of Contents

1. [Background: Audio-Visual Question Answering](#-background-audio-visual-question-answering)
2. [Model Overview](#-model-overview)
3. [Architecture Details](#-architecture-details)
4. [Project Layout](#-project-layout)
5. [Installation](#-installation)
6. [Data Preparation](#-data-preparation)
7. [Feature Extraction](#-feature-extraction)
8. [Training](#-training)
9. [Evaluation](#-evaluation)
10. [Smoke Test](#-smoke-test)
11. [Hyperparameters](#-hyperparameters)
12. [Benchmark Comparison](#-benchmark-comparison)
13. [Datasets in the Music AVQA Family](#-datasets-in-the-music-avqa-family)
14. [Acknowledgements](#-acknowledgements)
15. [Citation](#-citation)

---

## 🎼 Background: Audio-Visual Question Answering

**Audio-Visual Question Answering (AVQA)** requires a model to answer natural-language questions about a video by reasoning jointly over its audio and visual streams. **Music AVQA** is a particularly challenging subdomain because music performances exhibit:

| Challenge | Description |
| :--- | :--- |
| **Dense, continuous audio** | Overlapping polyphonic instruments with no silence between events |
| **Hierarchical temporal structure** | Information unfolds across beats, phrases, and sections |
| **Cross-modal correspondence** | Gesture-to-sound alignment with sub-second precision |
| **Domain-specific knowledge** | Instrument identity, rhythm, harmony, ensemble conventions |
| **Spatial-temporal reasoning** | *"Is the cello on the left more rhythmic than the one on the right?"* |

Standard AVQA models tuned for event-centric scenes (barking dogs, slamming doors) under-perform on music — motivating purpose-built architectures like **Spectra-AVQA**.

---

## 🏗️ Model Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  Frames ──► CLIP ViT-L/14  ──┐                                       │
│  Audio  ──► VGGish         ──┼──► TempTMP ──► TEMA ──► MS-CMAT ──►  │
│  Quest. ──► CLIP Text      ──┘   (Top-k    (per-mod)  (3-stage     │
│                                   + median             fusion)     MLP │
│                                   pool)                            ──► │
│                                                                Answer │
└─────────────────────────────────────────────────────────────────────┘
```

### Pipeline Stages

1. **Feature Embedding** — Frozen CLIP ViT-L/14 (frames + question text) and frozen VGGish (audio).
2. **TempTMP** (Temporal Top-k + Median Pooling) — Median-pool along time, then keep the `top_k` frames whose cosine similarity to the global question embedding is highest. Order-preserving. Applied to both video and audio.
3. **TEMA** (Transformer-based Encoding for Modality Adaptation) — Per-modality block: `LN → SRA-MHA → + → LN → MS-FFN → +`.
4. **MS-CMAT** (Multi-Stage Cross-Modality Attention) — Three pre-norm cross-attention stages fuse modalities progressively.
5. **Answering Head** — `concat(mean(O₁'), mean(O₂'), mean(O₃'))` → MLP classifier.

---

## 🔬 Architecture Details

### TEMA Block

- **SRA-MHA (Sparse-Region Attention MHA)** — Combines a local sliding window of size `sra_window` with `sra_global` always-visible global tokens. Works for both self-attention and cross-attention; substantially reduces compute on long sequences while preserving global context.
- **MS-FFN (Multi-Scale FFN)** — Parallel depthwise `Conv1d` branches with kernel sizes `{1, 3, 5}`, fused and projected back. Captures multi-resolution temporal patterns (onsets, motifs, phrases).

### MS-CMAT 3-Stage Fusion

| Stage | Query | Key / Value | Output |
| :---: | :---: | :---: | :---: |
| **1** | `F_q` (question) | `F_v` (visual) | `O₁'` = visual×question |
| **2** | `F_q` (question) | `F_a` (audio)  | `O₂'` = audio×question  |
| **3** | `O₁'`            | `O₂'`          | `O₃'` = joint reasoning |

Each stage uses **pre-LN + cross-attention + FFN**. The classifier consumes the mean-pooled output of *all three* stages, preserving uni- and cross-modal evidence.

---

## 📁 Project Layout

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

---

## ⚙️ Installation

```bash
# Clone the repo
git clone https://github.com/<your-username>/spectra-avqa.git
cd spectra-avqa

# Install Python dependencies
pip install -r requirements.txt

# VGGish weights (torchvggish downloads them on first run)
pip install git+https://github.com/harritaylor/torchvggish
```

> **Note:** `ffmpeg` must be on your `PATH` for frame and audio extraction.

---

## 📂 Data Preparation

```
data/
├── video/                              # raw .mp4s (MUSIC-AVQA or your own)
├── audio/                              # .wav extracted from video      (step 1)
├── frames/<vid>/                       # 1-fps JPGs per video           (step 2)
├── json/
│   ├── avqa-train.json                 # list of records
│   ├── avqa-val.json
│   ├── avqa-test.json
│   └── answer_vocab.json               # built automatically
└── feats/
    ├── clip_visual_frame/<vid>.npy     # (T, 768)
    ├── clip_visual_patch/<vid>.npy     # (T, P, 768)  [optional]
    ├── vggish_audio/<vid>.npy          # (T, 128)
    └── clip_question/<qid>.npy         # (1+Nq, 768)
```

### Record format (per JSON entry)

```json
{
  "video_id":    "00000001",
  "question_id": 12345,
  "question":    "Which instrument is used first?",
  "type":        "Audio-Visual",
  "answer":      "guitar"
}
```

> The loader also accepts the original MUSIC-AVQA typo `"anser"` for backwards compatibility.

---

## 🚀 Feature Extraction

Run the full feature-extraction pipeline in one command:

```bash
VIDEO_DIR=./data/video \
JSON_DIR=./data/json \
CLIP_MODEL=openai/clip-vit-large-patch14 \
bash scripts/extract_features.sh
```

This runs, in order:

1. `extract_audio.py` — wav per video
2. `extract_frames.py` — 1-fps JPGs per video
3. `extract_frames_ViT-L14.py` — CLIP frame features
4. `extract_vggish.py` — VGGish audio features
5. `extract_qst_ViT-L14.py` — CLIP question features
6. `build_answer_vocab.py` — answer vocabulary from train JSON
7. `verify_features.py` — sanity-checks every split has all its features

---

## 🏋️ Training

```bash
# Default config (configs/config.yaml)
bash scripts/train.sh

# With overrides
bash scripts/train.sh --batch_size 32 --lr 5e-5 --epochs 40 --top_k 10
```

Checkpoints are written to `./models/<checkpoint_name>/{last.pt, best.pt}`. The best checkpoint is selected by validation accuracy.

---

## 📊 Evaluation

```bash
# Default: evaluate on the test split
bash scripts/test.sh

# Evaluate on the validation split
SPLIT=val bash scripts/test.sh

# Evaluate a specific checkpoint
bash scripts/test.sh --ckpt ./models/my_run/best.pt
```

### Outputs

| File | Content |
| :--- | :--- |
| `results/<ckpt>/test_report.json` | Per-question-type and overall accuracy |
| `results/<ckpt>/test_preds.json`  | Per-sample predictions |

---

## 🧪 Smoke Test

Fast end-to-end sanity check that requires **no dataset**:

```bash
python scripts/sanity_check.py
```

This generates synthetic `.npy` features + JSON records in a temp directory and runs `main_train.py → main_test.py` end-to-end. Use it to verify your environment before downloading the full MUSIC-AVQA corpus.

---

## 🎛️ Hyperparameters

Defined in `configs/config.yaml`:

| Param | Default | Meaning |
| :--- | :---: | :--- |
| `dataset.num_frames`   | 60    | Frames per clip after uniform sampling |
| `dataset.top_k`        | 10    | `T′` retained by TempTMP |
| `model.d_model`        | 512   | Common projection dim |
| `model.sra_window`     | 8     | Local window size in SRA-MHA |
| `model.sra_global`     | 2     | Number of global tokens in SRA-MHA |
| `model.ms_ffn_kernels` | [1,3,5] | MS-FFN `Conv1d` kernel sizes |
| `model.n_layers`       | 2     | TEMA depth per modality |
| `train.lr`             | 1e-4  | AdamW learning rate |
| `train.amp`            | true  | fp16 mixed-precision |

---

## 🏆 Benchmark Comparison

The tables below place **Spectra-AVQA** in the broader landscape of Music AVQA methods. All numbers are top-1 accuracy (%) on the standard MUSIC-AVQA test split. *"–"* means the source did not report that cell.

### 📈 MUSIC-AVQA (v1) — Test Set

| Method | Venue / Year | A-Count | A-Comp | **A-Avg** | V-Count | V-Local | **V-Avg** | AV-Exist | AV-Count | AV-Local | AV-Comp | AV-Temp | **AV-Avg** | **Overall** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| FCNLSTM    | TASLP 2020   | 70.45 | 66.22 | 68.88 | – | – | – | 63.89 | 46.74 | 41.94 | 40.00 | 47.45 | 48.96 | 54.00 |
| CONVLSTM   | TASLP 2020   | 74.07 | 68.89 | 72.15 | – | – | – | 67.42 | 54.21 | 49.79 | 45.85 | 50.49 | 53.79 | 59.06 |
| GRU        | ICCV 2015    | 72.21 | 66.89 | 70.24 | 67.72 | 70.11 | 68.93 | 65.18 | 50.67 | 44.07 | 41.61 | 50.65 | 49.92 | 57.50 |
| HCAttn     | NeurIPS 2016 | 70.25 | 54.91 | 64.57 | 64.05 | 66.37 | 65.22 | 64.84 | 53.88 | 49.45 | 48.85 | 55.40 | 54.48 | 57.97 |
| MCAN       | CVPR 2019    | 77.50 | 55.24 | 69.25 | 71.56 | 70.93 | 71.24 | 80.40 | 54.48 | 55.05 | 43.59 | 52.42 | 57.20 | 64.18 |
| PSAC       | AAAI 2019    | 75.64 | 66.06 | 72.09 | 68.64 | 69.79 | 69.22 | 77.59 | 55.02 | 63.42 | 55.97 | 61.17 | 62.74 | 66.54 |
| HME        | CVPR 2019    | 74.76 | 63.56 | 70.61 | 67.97 | 69.46 | 68.76 | 80.30 | 53.18 | 46.45 | 45.61 | 51.28 | 55.72 | 63.19 |
| HCRN       | CVPR 2020    | 68.59 | 50.92 | 62.05 | 64.39 | 61.81 | 63.08 | 54.47 | 41.53 | 53.38 | 52.11 | 47.69 | 50.26 | 55.73 |
| AVSD       | CVPR 2019    | 72.41 | 61.90 | 68.52 | 67.39 | 74.19 | 70.83 | 81.61 | 58.79 | 63.89 | 61.52 | 61.41 | 65.49 | 67.44 |
| Pano-AVQA  | ICCV 2021    | 74.36 | 64.56 | 70.73 | 69.39 | 75.65 | 72.56 | 81.21 | 59.33 | 64.91 | 64.22 | 63.23 | 66.64 | 68.93 |
| **AVST**   | CVPR 2022    | 77.78 | 67.17 | 73.87 | 73.52 | 75.27 | 74.40 | 82.49 | 69.88 | 64.24 | 64.67 | 65.82 | 69.53 | 71.59 |
| COCA       | AAAI 2023    | 79.35 | 67.68 | 75.42 | 75.10 | 75.43 | 75.23 | 83.50 | 66.63 | 69.72 | 64.12 | 65.57 | 69.96 | 72.33 |
| PSTP-Net   | ACM MM 2023  | 73.97 | 65.59 | 70.91 | 77.15 | 77.36 | 77.26 | 76.18 | 72.23 | 71.80 | 71.79 | 69.00 | 72.57 | 73.52 |
| LAVisH     | CVPR 2023    | 82.09 | 65.56 | 75.97 | 78.98 | 81.43 | 80.22 | 81.71 | 75.51 | 66.13 | 63.77 | 67.96 | 71.26 | 74.46 |
| LSTTA      | ACM MM 2023  | 81.75 | 82.04 | 81.90 | 81.82 | 82.23 | 82.03 | 83.46 | 79.11 | 78.23 | 78.02 | 79.32 | 79.63 | 81.19 |
| DG-SCT     | NeurIPS 2023 | 83.27 | 64.56 | 76.34 | 81.57 | 82.57 | 82.08 | 81.61 | 72.84 | 65.91 | 64.22 | 67.48 | 70.56 | 74.62 |
| SaSR-Net   | arXiv 2024   | 77.83 | 67.59 | 73.06 | 75.84 | 76.13 | 74.83 | 81.49 | 73.43 | 73.64 | 79.15 | 77.46 | 74.66 | 74.21 |
| LAST-Att   | WACV 2024    | 85.71 | 63.10 | –     | 83.86 | 83.09 | –     | 76.47 | 76.20 | 68.91 | 65.60 | 66.75 | –     | 75.45 |
| APL        | AAAI 2024    | 82.40 | 70.71 | 78.09 | 76.52 | 82.74 | 79.69 | 82.99 | 73.29 | 66.68 | 64.76 | 65.95 | 70.96 | 74.53 |
| TSPM       | ACM MM 2024  | 84.07 | 64.65 | 76.91 | 82.29 | 84.90 | 83.61 | 82.19 | 76.21 | 71.85 | 65.76 | 71.17 | 73.51 | 76.79 |
| MCCD       | NeurIPS 2024 | 83.87 | 71.04 | 79.14 | 79.78 | 76.73 | 78.24 | 80.87 | 71.46 | 51.63 | 64.67 | 64.60 | 67.13 | 72.20 |
| Amuse      | EMNLP 2024   | 84.61 | 82.45 | 83.58 | 87.14 | 84.39 | 85.84 | 86.95 | 85.49 | 73.01 | 82.98 | 83.06 | 82.43 | 83.52 |
| Sparsify   | ACL 2025     | 83.12 | 77.64 | 80.38 | 83.12 | 85.74 | 84.43 | 80.98 | 82.70 | 85.09 | 77.12 | 79.89 | 81.80 | 81.75 |
| PSOT       | AAAI 2025    | –     | –     | 78.22 | –     | –     | 80.07 | –     | –     | –     | –     | –     | 72.61 | 75.29 |
| QA-TIGER   | CVPR 2025    | 84.86 | 67.85 | 78.58 | 83.96 | 86.29 | 85.14 | 83.10 | 78.58 | 72.50 | 63.94 | 69.59 | 73.74 | 77.62 |
| QSTar      | arXiv 2026   | 85.64 | 72.05 | 80.63 | 83.46 | 84.90 | 84.17 | 83.81 | 79.76 | 72.72 | 70.03 | 72.38 | 75.98 | 78.98 |
| AV-Master  | arXiv 2025   | –     | –     | 79.95 | –     | –     | 86.58 | –     | –     | –     | –     | –     | 74.22 | 78.51 |
| **Spectra-AVQA (Ours)** | *Under Review* | — | — | — | — | — | — | — | — | — | — | — | — | — |

> 📝 *Spectra-AVQA results are pending official publication; this README will be updated upon acceptance at IEEE TCSS.*

### 🔁 Large Language Model Baselines (Zero-Shot)

| Method | Venue / Year | Overall Accuracy (%) |
| :--- | :---: | :---: |
| OneLLM     | CVPR 2024 | 43.0 |
| CAT        | ECCV 2024 | 48.6 |
| CAT+       | ECCV 2024 | 50.1 |
| AVicuna    | 2024      | 49.6 |

Pre-trained MLLMs lag substantially behind purpose-built AVQA models, reinforcing the value of **specialized architectures** like Spectra-AVQA for music understanding.

### 🎯 MUSIC-AVQA v2.0 (Rebalanced)

| Method | Venue / Year | A-Avg | V-Avg | AV-Avg | **Overall** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| AVST     | CVPR 2022    | 75.20 | 78.05 | 65.83 | 70.83 |
| LAVisH   | CVPR 2023    | 75.72 | 82.30 | 67.75 | 73.28 |
| DG-SCT   | NeurIPS 2023 | 73.64 | 82.48 | 70.38 | 74.08 |
| LAST-Att | WACV 2024    | –     | –     | –     | 75.44 |
| Amuse    | EMNLP 2024   | 84.34 | 86.74 | 85.51 | 85.16 |

### 🛡️ MUSIC-AVQA-R (Robustness / Rephrased)

| Method | Venue / Year | Overall (%) |
| :--- | :---: | :---: |
| AVST     | CVPR 2022    | 63.89 |
| LAVisH   | CVPR 2023    | 65.00 |
| DG-SCT   | NeurIPS 2023 | 66.50 |
| PSTP-Net | ACM MM 2023  | 64.50 |
| MCCD     | NeurIPS 2024 | 67.59 |
| QA-TIGER | CVPR 2025    | 67.99 |

---

## 🗂️ Datasets in the Music AVQA Family

| Dataset | Year | Videos | QA Pairs | Purpose |
| :--- | :---: | :---: | :---: | :--- |
| **MUSIC-AVQA**     | CVPR 2022    | 9,288  | 45,867  | Core benchmark, 22 instruments, 33 templates |
| **MUSIC-AVQA v2.0**| WACV 2024    | 10,518 | ~54,000 | Rebalanced (no dominant answer > 60% / 50%) |
| **MUSIC-AVQA-R**   | NeurIPS 2024 | 9,288  | 211,572 | Robustness via question rephrasing (head/tail) |
| **FortisAVQA**     | 2025         | —      | —       | Two-stage robustness + distribution shifts |

### Question Type Glossary

| Code | Meaning |
| :--- | :--- |
| **A-Count**  | *"How many instruments are sounding in the video?"* |
| **A-Comp**   | *"Is instrument A louder than instrument B?"* |
| **V-Count**  | *"How many instruments are visible in the video?"* |
| **V-Local**  | *"Where is instrument X located?"* |
| **AV-Exist** | *"Is the sounding object visible?"* |
| **AV-Count** | *"How many instruments are simultaneously sounding & visible?"* |
| **AV-Local** | *"Locate the sounding instrument in the video."* |
| **AV-Comp**  | *"Is the cello on the left more rhythmic than the one on the right?"* |
| **AV-Temp**  | *"Which instrument plays first / after?"* |

---

## 🙏 Acknowledgements

This project stands on the shoulders of outstanding open-source contributions:

- **[MUSIC-AVQA](https://github.com/GeWu-Lab/MUSIC-AVQA)** (Li et al., CVPR 2022) — dataset and pipeline conventions
- **[TSPM](https://github.com/GeWu-Lab/TSPM)** (Li et al., ACM MM 2024) — temporal selection and question-guided attention ideas
- **[CLIP](https://github.com/openai/CLIP)** (Radford et al.) — frozen visual + text encoder
- **[VGGish](https://github.com/harritaylor/torchvggish)** (Hershey et al.) — frozen audio encoder

We also thank the authors of LAVISH, DG-SCT, PSTP-Net, Amuse, QA-TIGER, and QSTar for releasing code/results that enabled the comparisons in this README.

---

## 📚 Citation

If you find this repository or the benchmark comparison **helpful in your research**, please consider citing our paper:

```bibtex
@article{spectra_avqa_2025,
  title   = {SPECTRA-AVQA: Sparse-Region Attention and Multi-Stage Cross-Modality
             Transformer for Music Audio-Visual Question Answering},
  author  = {<Author List>},
  journal = {IEEE Transactions on Computational Social Systems (Under Review)},
  year    = {2025},
  note    = {Manuscript ID: TCSS-2025-07-1381.R2}
}
```

> 💡 *Please also cite the original MUSIC-AVQA paper and any baselines referenced in the comparison tables above when reporting results from this codebase.*

---

<div align="center">

### ⭐ If you found this page helpful, please consider citing our paper and starring the repository!

**Made with ❤️ for the Music AI community**

</div>
