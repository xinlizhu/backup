#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
benchmark_root="$(dirname "${project_root}")"
workspace_root="$(dirname "${benchmark_root}")"
output_root="${project_root}/outputs/opd_student_teacher_matrix"
qwen8_path="${workspace_root}/CIR_PLAY/models/Qwen3-VL-Embedding-8B"
qwen2_path="${workspace_root}/CIR_PLAY/models/Qwen3-VL-Embedding-2B"
s2_cache="${project_root}/outputs/on_policy_centered/cache/bge_mllm_s2_cirr_train"
fashioniq_root="${FASHIONIQ_ROOT:-${workspace_root}/CIR_new/datasets/fashion-iq}"

clip_names=(
  CLIP-B-OPD-S2
  CLIP-B-OPD-Qwen8
  CLIP-L-OPD-Qwen8
)
clip_runs=(
  clip_b_from_s2
  clip_b_from_qwen8
  clip_l_from_qwen8
)
clip_configs=(
  "${project_root}/configs/opd_clip_b_s2_rank.yaml"
  "${project_root}/configs/opd_clip_b_qwen8_rank.yaml"
  "${project_root}/configs/opd_clip_l_qwen8_rank.yaml"
)
qwen_name="Qwen2-QueryLoRA-OPD-Qwen8"
qwen_config="${project_root}/configs/opd_qwen2_qwen8_rank.yaml"

for required in \
  "${qwen8_path}/config.json" \
  "${qwen2_path}/config.json" \
  "${s2_cache}/metadata.json"; do
  if [[ ! -f "${required}" ]]; then
    echo "[error] missing required artifact: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${output_root}"
cd "${project_root}"

echo "[stage 1/8] cache Qwen3-VL-Embedding-8B Teacher on CIRR train"
"${PYTHON}" scripts/prepare_on_policy_teacher_cache.py \
  --config "${clip_configs[1]}" \
  --teacher-model-name Qwen3-VL-Embedding-8B \
  --teacher-model-path "${qwen8_path}" \
  --device cuda:0 \
  --image-batch-size "${QWEN8_TEACHER_IMAGE_BATCH_SIZE:-4}" \
  --query-batch-size "${QWEN8_TEACHER_QUERY_BATCH_SIZE:-4}"

echo "[stage 2/8] cache immutable Qwen-2B Student gallery"
"${PYTHON}" scripts/prepare_qwen_student_gallery_cache.py \
  --config "${qwen_config}" \
  --device cuda:0 \
  --image-batch-size "${QWEN2_GALLERY_BATCH_SIZE:-16}"

echo "[stage 3/8] train three CLIP Student/Teacher cells"
for index in "${!clip_runs[@]}"; do
  run_dir="${output_root}/${clip_runs[$index]}"
  final_checkpoint="${run_dir}/checkpoint_epoch_004.pt"
  if [[ -f "${final_checkpoint}" ]]; then
    echo "[skip train] ${clip_names[$index]}: ${final_checkpoint}"
  elif [[ -f "${run_dir}/last.pt" ]]; then
    echo "[resume train] ${clip_names[$index]}"
    "${PYTHON}" scripts/train_on_policy_distillation.py \
      --config "${clip_configs[$index]}" \
      --resume-from "${run_dir}/last.pt"
  else
    echo "[train] ${clip_names[$index]}"
    "${PYTHON}" scripts/train_on_policy_distillation.py \
      --config "${clip_configs[$index]}"
  fi
done

echo "[stage 4/8] train Qwen-2B query-LoRA Student from Qwen-8B"
qwen_run_dir="${output_root}/qwen2_from_qwen8"
qwen_checkpoint="${qwen_run_dir}/checkpoint_epoch_004.pt"
if [[ -f "${qwen_checkpoint}" ]]; then
  echo "[skip train] ${qwen_name}: ${qwen_checkpoint}"
elif [[ -f "${qwen_run_dir}/last.pt" ]]; then
  "${PYTHON}" scripts/train_qwen_on_policy_distillation.py \
    --config "${qwen_config}" \
    --resume-from "${qwen_run_dir}/last.pt"
else
  "${PYTHON}" scripts/train_qwen_on_policy_distillation.py \
    --config "${qwen_config}"
fi

echo "[stage 5/8] CIRR val for the raw CLIP-B control and all new Students"
clip_b_raw_metrics="${output_root}/clip_b_original/cirr_val.json"
mkdir -p "$(dirname "${clip_b_raw_metrics}")"
if [[ ! -f "${clip_b_raw_metrics}" ]]; then
  "${PYTHON}" scripts/evaluate_on_policy_distillation.py \
    --config "${clip_configs[0]}" \
    --output "${clip_b_raw_metrics}"
fi
for index in "${!clip_runs[@]}"; do
  metrics="${output_root}/${clip_runs[$index]}/val_epoch_004.json"
  if [[ ! -f "${metrics}" ]]; then
    "${PYTHON}" scripts/evaluate_on_policy_distillation.py \
      --config "${clip_configs[$index]}" \
      --checkpoint "${output_root}/${clip_runs[$index]}/checkpoint_epoch_004.pt" \
      --output "${metrics}"
  fi
done
if [[ ! -f "${qwen_run_dir}/cirr_val/val_metrics.json" ]]; then
  "${PYTHON}" scripts/evaluate_cirr_original_adapter.py \
    --config "${qwen_config}" \
    --model-name "${qwen_name}" \
    --model-path "${qwen_checkpoint}" \
    --output-dir "${qwen_run_dir}/cirr_val" \
    --device cuda:0 \
    --dtype auto \
    --image-batch-size "${QWEN2_EVAL_BATCH_SIZE:-16}" \
    --query-batch-size "${QWEN2_EVAL_BATCH_SIZE:-16}"
fi

if [[ "${RUN_EXTENDED_EVAL:-1}" == "1" ]]; then
  clip_checkpoint_args=()
  clip_model_csv="CLIP-B-original"
  for index in "${!clip_runs[@]}"; do
    checkpoint="${output_root}/${clip_runs[$index]}/checkpoint_epoch_004.pt"
    clip_checkpoint_args+=(--checkpoint "${clip_names[$index]}=${checkpoint}")
    clip_model_csv+=",${clip_names[$index]}"
  done

  echo "[stage 6/8] official PinPoint mAP@10 and NegRecall@10"
  (
    cd "${benchmark_root}/diag_model"
    "${PYTHON}" run_clip_pinpoint_official.py \
      --models "${clip_model_csv}" \
      --pretrained-clip "CLIP-B-original=${project_root}/models/clip-vit-base-patch16" \
      "${clip_checkpoint_args[@]}" \
      --output-root "${output_root}/pinpoint_official" \
      --device cuda:0 \
      --batch-size "${CLIP_EVAL_BATCH_SIZE:-128}"
    "${PYTHON}" run_clip_pinpoint_official.py \
      --models "${qwen_name}" \
      --adapter "${qwen_name}=${qwen_checkpoint}" \
      --output-root "${output_root}/pinpoint_official" \
      --shared-corpus-models "${qwen_name}" \
      --frozen-corpus-cache "${project_root}/outputs/qwen3_vl_original_baselines/pinpoint_official/Qwen3-VL-Embedding-2B/cache/corpus" \
      --device cuda:0 \
      --batch-size "${QWEN2_EVAL_BATCH_SIZE:-16}"
  )

  echo "[stage 7/8] CIR-QDiag Delta, raw/stable win and random-10 separation"
  (
    cd "${benchmark_root}/diag_model"
    "${PYTHON}" evaluate_winrate_random10.py \
      --pretrained-clip "CLIP-B-original=${project_root}/models/clip-vit-base-patch16" \
      "${clip_checkpoint_args[@]}" \
      --adapter "${qwen_name}=${qwen_checkpoint}" \
      --output-root "${output_root}/qdiag" \
      --device cuda:0 \
      --batch-size "${QDIAG_BATCH_SIZE:-16}" \
      --random-samples 10 \
      --seed 42
  )

  echo "[stage 8/8] FashionIQ transfer"
  clip_b_fiq="${output_root}/fashioniq/clip_b_original"
  if [[ ! -f "${clip_b_fiq}/metrics.json" ]]; then
    "${PYTHON}" scripts/evaluate_fashioniq_clip.py \
      --model-name CLIP-B-original \
      --config "${clip_configs[0]}" \
      --dataset-root "${fashioniq_root}" \
      --output-dir "${clip_b_fiq}" \
      --device cuda:0
  fi
  for index in "${!clip_runs[@]}"; do
    model_output="${output_root}/fashioniq/${clip_runs[$index]}"
    if [[ ! -f "${model_output}/metrics.json" ]]; then
      "${PYTHON}" scripts/evaluate_fashioniq_clip.py \
        --model-name "${clip_names[$index]}" \
        --config "${clip_configs[$index]}" \
        --checkpoint "${output_root}/${clip_runs[$index]}/checkpoint_epoch_004.pt" \
        --dataset-root "${fashioniq_root}" \
        --output-dir "${model_output}" \
        --device cuda:0
    fi
  done
  if [[ ! -f "${output_root}/fashioniq/qwen2_from_qwen8/metrics.json" ]]; then
    "${PYTHON}" scripts/evaluate_fashioniq_original_adapter.py \
      --model-name "${qwen_name}" \
      --adapter-name "${qwen_name}" \
      --model-path "${qwen_checkpoint}" \
      --dataset-root "${fashioniq_root}" \
      --output-dir "${output_root}/fashioniq/qwen2_from_qwen8" \
      --device cuda:0 \
      --image-batch-size "${QWEN2_EVAL_BATCH_SIZE:-16}" \
      --query-batch-size "${QWEN2_EVAL_BATCH_SIZE:-16}"
  fi
else
  echo "[skip stages 6-8] RUN_EXTENDED_EVAL=${RUN_EXTENDED_EVAL}"
fi

"${PYTHON}" scripts/summarize_opd_student_teacher_matrix.py \
  --output-root "${output_root}"

echo "[done] ${output_root}/RESULTS.md"
