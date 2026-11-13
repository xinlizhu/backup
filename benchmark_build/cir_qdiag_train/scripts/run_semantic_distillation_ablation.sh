#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
benchmark_root="$(dirname "${project_root}")"
config_dir="${project_root}/configs"
output_root="${project_root}/outputs/semantic_distillation/clip_l_full_baseline"
baseline_checkpoint="${project_root}/outputs/clip_l/full/baseline/checkpoint_epoch_010.pt"
teacher_model_path="${TEACHER_MODEL_PATH:-/mnt/data/datasets/search_dataset/zxl/CIR_PLAY/models/checkpoints/BGE-VL-MLLM-S2}"

if [[ ! -f "${baseline_checkpoint}" ]]; then
  echo "Missing Student initialization checkpoint: ${baseline_checkpoint}" >&2
  exit 1
fi
if [[ ! -d "${teacher_model_path}" ]]; then
  echo "Missing Teacher model: ${teacher_model_path}" >&2
  exit 1
fi

cd "${project_root}"

rank_config="${config_dir}/semantic_distill_clip_l_positive_rank.yaml"

echo "[stage 1/5] cache frozen BGE-VL-MLLM-S2 signals and CLIP-L images"
"${PYTHON}" scripts/prepare_semantic_distillation_cache.py \
  --config "${rank_config}" \
  --teacher-model-name BGE-VL-MLLM-S2 \
  --teacher-model-path "${teacher_model_path}" \
  --device cuda:0 \
  --teacher-batch-size "${TEACHER_BATCH_SIZE:-16}" \
  --student-image-batch-size "${STUDENT_IMAGE_BATCH_SIZE:-128}"

run_names=(positive_rank semantic_response joint)
configs=(
  "${config_dir}/semantic_distill_clip_l_positive_rank.yaml"
  "${config_dir}/semantic_distill_clip_l_response.yaml"
  "${config_dir}/semantic_distill_clip_l_joint.yaml"
)

echo "[stage 2/5] train three KD variants for 16 epochs"
for index in "${!run_names[@]}"; do
  run_name="${run_names[$index]}"
  config="${configs[$index]}"
  run_dir="${output_root}/${run_name}"
  final_checkpoint="${run_dir}/checkpoint_epoch_016.pt"
  if [[ -f "${final_checkpoint}" ]]; then
    echo "[skip train] ${run_name}: ${final_checkpoint}"
  elif [[ -f "${run_dir}/last.pt" ]]; then
    echo "[resume train] ${run_name}: ${run_dir}/last.pt"
    "${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=1 \
      scripts/train_semantic_distillation.py \
      --config "${config}" \
      --resume-from "${run_dir}/last.pt"
  else
    echo "[train] ${run_name}"
    "${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=1 \
      scripts/train_semantic_distillation.py \
      --config "${config}"
  fi
done

echo "[stage 3/5] evaluate CIRR val and Group Recall"
baseline_metrics="${output_root}/baseline/val_metrics.json"
mkdir -p "$(dirname "${baseline_metrics}")"
if [[ ! -f "${baseline_metrics}" ]]; then
  "${PYTHON}" scripts/evaluate_semantic_distillation.py \
    --config "${rank_config}" \
    --checkpoint "${baseline_checkpoint}" \
    --output "${baseline_metrics}"
fi

if [[ "${EVAL_CURVE:-1}" == "1" ]]; then
  eval_epochs=(004 008 012 016)
else
  eval_epochs=(016)
fi
for index in "${!run_names[@]}"; do
  run_name="${run_names[$index]}"
  config="${configs[$index]}"
  for epoch in "${eval_epochs[@]}"; do
    checkpoint="${output_root}/${run_name}/checkpoint_epoch_${epoch}.pt"
    metrics="${output_root}/${run_name}/val_epoch_${epoch}.json"
    if [[ ! -f "${metrics}" ]]; then
      "${PYTHON}" scripts/evaluate_semantic_distillation.py \
        --config "${config}" \
        --checkpoint "${checkpoint}" \
        --output "${metrics}"
    fi
  done
done

model_names="CLIP-L-KD-baseline,CLIP-L-KD-rank,CLIP-L-KD-response,CLIP-L-KD-joint"
checkpoint_args=(
  --checkpoint "CLIP-L-KD-baseline=${baseline_checkpoint}"
  --checkpoint "CLIP-L-KD-rank=${output_root}/positive_rank/checkpoint_epoch_016.pt"
  --checkpoint "CLIP-L-KD-response=${output_root}/semantic_response/checkpoint_epoch_016.pt"
  --checkpoint "CLIP-L-KD-joint=${output_root}/joint/checkpoint_epoch_016.pt"
)

echo "[stage 4/5] evaluate official PinPoint mAP@10 and NegRecall@10"
if [[ "${RUN_PINPOINT_OFFICIAL:-1}" == "1" ]]; then
  (
    cd "${benchmark_root}/diag_model"
    "${PYTHON}" run_clip_pinpoint_official.py \
      --models "${model_names}" \
      "${checkpoint_args[@]}" \
      --shared-corpus-models "${model_names}" \
      --output-root "${output_root}/pinpoint_official" \
      --device cuda:0 \
      --batch-size "${PINPOINT_BATCH_SIZE:-64}"
  )
fi

echo "[stage 5/5] evaluate PinPoint Delta, calibrated Stable Win, and random-10 gap"
if [[ "${RUN_PINPOINT_DIAGNOSTICS:-1}" == "1" ]]; then
  (
    cd "${benchmark_root}/diag_model"
    "${PYTHON}" evaluate_winrate_random10.py \
      "${checkpoint_args[@]}" \
      --output-root "${output_root}/pinpoint_diagnostics" \
      --device cuda:0 \
      --batch-size "${PINPOINT_BATCH_SIZE:-64}" \
      --random-samples 10
  )
fi

"${PYTHON}" scripts/summarize_semantic_distillation_ablation.py \
  --output-root "${output_root}"

echo "[done] ${output_root}/ABLATION_SUMMARY.md"
