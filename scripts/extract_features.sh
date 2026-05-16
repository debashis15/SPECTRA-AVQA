#!/usr/bin/env bash
# scripts/extract_features.sh
# -----------------------------------------------------------------
# Runs the whole feature-extraction pipeline end-to-end.
# -----------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VIDEO_DIR="${VIDEO_DIR:-./data/video}"
AUDIO_DIR="${AUDIO_DIR:-./data/audio}"
FRAME_DIR="${FRAME_DIR:-./data/frames}"
CLIP_FRAME_DIR="${CLIP_FRAME_DIR:-./data/feats/clip_visual_frame}"
CLIP_PATCH_DIR="${CLIP_PATCH_DIR:-./data/feats/clip_visual_patch}"
VGGISH_DIR="${VGGISH_DIR:-./data/feats/vggish_audio}"
QST_FEAT_DIR="${QST_FEAT_DIR:-./data/feats/clip_question}"
JSON_DIR="${JSON_DIR:-./data/json}"
CLIP_MODEL="${CLIP_MODEL:-openai/clip-vit-large-patch14}"

echo "============================================================"
echo "[1/5]  extract audio from videos"
echo "============================================================"
python feat_script/extract_audio_cues/extract_audio.py \
    --video_dir "$VIDEO_DIR" \
    --audio_dir "$AUDIO_DIR"

echo "============================================================"
echo "[2/5]  extract frames (1 fps)"
echo "============================================================"
python feat_script/extract_visual_frames/extract_frames.py \
    --video_dir "$VIDEO_DIR" \
    --frame_dir "$FRAME_DIR" \
    --fps 1

echo "============================================================"
echo "[3/5]  CLIP visual features (frame-level)"
echo "============================================================"
python feat_script/extract_clip_feat/extract_frames_ViT-L14.py \
    --frame_dir      "$FRAME_DIR" \
    --clip_frame_dir "$CLIP_FRAME_DIR" \
    --clip_patch_dir "$CLIP_PATCH_DIR" \
    --clip_model     "$CLIP_MODEL" \
    --batch_size 32

echo "============================================================"
echo "[4/5]  VGGish audio features"
echo "============================================================"
python feat_script/extract_vggish_feat/extract_vggish.py \
    --audio_dir   "$AUDIO_DIR" \
    --vggish_dir  "$VGGISH_DIR"

echo "============================================================"
echo "[5/5]  CLIP question features + answer vocab"
echo "============================================================"
python feat_script/extract_clip_feat/extract_qst_ViT-L14.py \
    --json_files "$JSON_DIR/avqa-train.json" \
                 "$JSON_DIR/avqa-val.json" \
                 "$JSON_DIR/avqa-test.json" \
    --qst_feat_dir "$QST_FEAT_DIR" \
    --clip_model   "$CLIP_MODEL"

python feat_script/build_answer_vocab.py \
    --json_files "$JSON_DIR/avqa-train.json" \
    --out        "$JSON_DIR/answer_vocab.json"

echo "============================================================"
echo "[+] verify features"
echo "============================================================"
for split in train val test; do
  python feat_script/verify_features.py \
      --json_file      "$JSON_DIR/avqa-${split}.json" \
      --clip_frame_dir "$CLIP_FRAME_DIR" \
      --vggish_dir     "$VGGISH_DIR" \
      --qst_feat_dir   "$QST_FEAT_DIR" \
      --report         "./logs/missing_${split}.txt"
done

echo "All done."
