#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG="configs/on_policy_clip_l_rank.yaml"
OUTPUT_ROOT="outputs/cirr_train_recall"
TEACHER_CACHE="outputs/on_policy_centered/cache/bge_mllm_s2_cirr_train"
CLIP_CACHE="${OUTPUT_ROOT}/clip_l_initial_student"
CLIP_GALLERY="outputs/on_policy_centered/clip_l_original/positive_rank/on_policy/epoch_001/gallery_features.npy"
BGE_LARGE_CACHE="${OUTPUT_ROOT}/bge_vl_large_original"
DEVICE="${DEVICE:-cuda:0}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-512}"
BGE_IMAGE_BATCH_SIZE="${BGE_IMAGE_BATCH_SIZE:-512}"
BGE_QUERY_BATCH_SIZE="${BGE_QUERY_BATCH_SIZE:-1024}"
CLIP_QUERY_BATCH_SIZE="${CLIP_QUERY_BATCH_SIZE:-1024}"

echo "[1/6] evaluate cached BGE-VL-MLLM-S2 Teacher"
python scripts/evaluate_cirr_train_cache.py \
  --config "${CONFIG}" \
  --cache-dir "${TEACHER_CACHE}" \
  --model-label "BGE-VL-MLLM-S2 Teacher" \
  --device "${DEVICE}" \
  --score-batch-size "${SCORE_BATCH_SIZE}"

echo "[2/6] reconstruct/cache initial CLIP-L Student train queries"
python scripts/prepare_original_clip_student_train_cache.py \
  --config "${CONFIG}" \
  --gallery-features "${CLIP_GALLERY}" \
  --output-dir "${CLIP_CACHE}" \
  --device "${DEVICE}" \
  --query-batch-size "${CLIP_QUERY_BATCH_SIZE}"

echo "[3/6] evaluate initial CLIP-L Student"
python scripts/evaluate_cirr_train_cache.py \
  --config "${CONFIG}" \
  --cache-dir "${CLIP_CACHE}" \
  --gallery-features "${CLIP_GALLERY}" \
  --model-label "CLIP-L initial Student" \
  --device "${DEVICE}" \
  --score-batch-size "${SCORE_BATCH_SIZE}"

echo "[4/6] cache original BGE-VL-large train gallery and queries"
python scripts/prepare_on_policy_teacher_cache.py \
  --config "${CONFIG}" \
  --teacher-model-name "BGE-VL-large" \
  --teacher-model-path "/mnt/data/datasets/search_dataset/zxl/CIR_PLAY/models/checkpoints/BGE-VL-large" \
  --output-dir "${BGE_LARGE_CACHE}" \
  --device "${DEVICE}" \
  --dtype auto \
  --image-batch-size "${BGE_IMAGE_BATCH_SIZE}" \
  --query-batch-size "${BGE_QUERY_BATCH_SIZE}"

echo "[5/6] evaluate original BGE-VL-large"
python scripts/evaluate_cirr_train_cache.py \
  --config "${CONFIG}" \
  --cache-dir "${BGE_LARGE_CACHE}" \
  --model-label "BGE-VL-large original" \
  --device "${DEVICE}" \
  --score-batch-size "${SCORE_BATCH_SIZE}"

echo "[6/6] write combined report"
python scripts/summarize_cirr_train_recall.py \
  --result "${TEACHER_CACHE}/train_recall.json" \
  --result "${CLIP_CACHE}/train_recall.json" \
  --result "${BGE_LARGE_CACHE}/train_recall.json" \
  --output-dir "${OUTPUT_ROOT}"

echo "Done: ${PROJECT_ROOT}/${OUTPUT_ROOT}/summary.md"
