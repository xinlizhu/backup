#!/usr/bin/env bash
set -euo pipefail

# Each model runs a matched continuation control and the typed-candidate method.
# Start with MODELS=clip_l. Add BGE only after checking the CLIP-L result.
MODELS="${MODELS:-clip_l}"
RUNS="${RUNS:-control typed}"
PYTHON="${PYTHON:-python}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
benchmark_root="$(dirname "${project_root}")"

for model in ${MODELS}; do
  case "${model}" in
    clip_l)
      initialize_from="outputs/clip_l/full/hard_w0_1/checkpoint_epoch_010.pt"
      ;;
    bge_vl_large)
      initialize_from="outputs/bge_vl_large/full/hard_w0_1/checkpoint_epoch_010.pt"
      ;;
    *)
      echo "Unsupported model: ${model}; expected clip_l or bge_vl_large" >&2
      exit 2
      ;;
  esac

  if [[ ! -f "${initialize_from}" ]]; then
    echo "Missing initialization checkpoint: ${initialize_from}" >&2
    exit 1
  fi

  for run in ${RUNS}; do
    case "${model}:${run}" in
      clip_l:control)
        config="configs/clip_l_full_hard_w0_1_continue_v5.yaml"
        output_dir="outputs/clip_l/full/hard_w0_1_continue_v5"
        ;;
      clip_l:typed)
        config="configs/clip_l_full_typed_candidate_w0_1.yaml"
        output_dir="outputs/clip_l/full/typed_candidate_w0_1"
        ;;
      bge_vl_large:control)
        config="configs/bge_vl_large_full_hard_w0_1_continue_v5.yaml"
        output_dir="outputs/bge_vl_large/full/hard_w0_1_continue_v5"
        ;;
      bge_vl_large:typed)
        config="configs/bge_vl_large_full_typed_candidate_w0_1.yaml"
        output_dir="outputs/bge_vl_large/full/typed_candidate_w0_1"
        ;;
      *)
        echo "Unsupported run: ${model}:${run}" >&2
        exit 2
        ;;
    esac

    final_checkpoint="${output_dir}/checkpoint_epoch_003.pt"
    if [[ -f "${final_checkpoint}" ]]; then
      echo "[skip train] ${final_checkpoint}"
    elif [[ -f "${output_dir}/last.pt" ]]; then
      echo "[resume train] ${output_dir}/last.pt"
      "${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=1 \
        scripts/train.py \
        --config "${config}" \
        --resume-from "${output_dir}/last.pt"
    else
      echo "[train] ${model}:${run}"
      "${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=1 \
        scripts/train.py \
        --config "${config}" \
        --initialize-from "${initialize_from}"
    fi

    "${PYTHON}" scripts/evaluate.py \
      --config "${config}" \
      --checkpoint "${final_checkpoint}" \
      --output "${output_dir}/val_metrics.json"
    "${PYTHON}" scripts/evaluate_reverse.py \
      --config "${config}" \
      --checkpoint "${final_checkpoint}" \
      --output "${output_dir}/reverse_val_metrics.json"
  done
done

if [[ "${RUN_PINPOINT:-0}" == "1" ]]; then
  model_names=()
  checkpoint_args=()
  for model in ${MODELS}; do
    for run in ${RUNS}; do
      case "${model}:${run}" in
        clip_l:control)
          name="CLIP-L-hard-w0.1-v5-control"
          checkpoint="${project_root}/outputs/clip_l/full/hard_w0_1_continue_v5/checkpoint_epoch_003.pt"
          ;;
        clip_l:typed)
          name="CLIP-L-typed-candidate-w0.1"
          checkpoint="${project_root}/outputs/clip_l/full/typed_candidate_w0_1/checkpoint_epoch_003.pt"
          ;;
        bge_vl_large:control)
          name="BGE-VL-large-hard-w0.1-v5-control"
          checkpoint="${project_root}/outputs/bge_vl_large/full/hard_w0_1_continue_v5/checkpoint_epoch_003.pt"
          ;;
        bge_vl_large:typed)
          name="BGE-VL-large-typed-candidate-w0.1"
          checkpoint="${project_root}/outputs/bge_vl_large/full/typed_candidate_w0_1/checkpoint_epoch_003.pt"
          ;;
      esac
      model_names+=("${name}")
      checkpoint_args+=(--checkpoint "${name}=${checkpoint}")
    done
  done
  models_csv="$(IFS=,; echo "${model_names[*]}")"
  (
    cd "${benchmark_root}/diag_model"
    "${PYTHON}" run_clip_pinpoint_official.py \
      --models "${models_csv}" \
      "${checkpoint_args[@]}" \
      --output-root "${project_root}/outputs/typed_candidate_pinpoint_official" \
      --batch-size "${PINPOINT_BATCH_SIZE:-64}"
  )
fi
