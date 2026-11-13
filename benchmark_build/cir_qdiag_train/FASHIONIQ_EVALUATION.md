# FashionIQ validation evaluation

## Protocol

- Split: FashionIQ `val`.
- Categories: `dress`, `shirt`, and `toptee`.
- Query text: strip `.?, ` from both provided captions and concatenate them in
  stored order as `caption_1 and caption_2`.
- Gallery: the category-specific FashionIQ validation split.
- The reference image remains in the gallery, matching the existing local
  FashionIQ evaluator.
- Standard reported metrics: Recall@10 and Recall@50 per category, followed by
  the unweighted macro average over the three categories.
- Per-model JSON also records Recall@1, Recall@5, and query-weighted micro
  averages.

The released validation data contains 6,016 queries and 15,536 category-gallery
entries in total:

| Category | Queries | Gallery |
|---|---:|---:|
| dress | 2,017 | 3,817 |
| shirt | 2,038 | 6,346 |
| toptee | 1,961 | 5,373 |

## Models

The runner evaluates these models sequentially on one GPU:

1. Original CLIP-L (zero training).
2. CLIP-L Full InfoNCE baseline.
3. CLIP-L On-Policy Positive Rank KD.
4. CLIP-L On-Policy Centered Response KD.
5. CLIP-L On-Policy Joint KD.
6. BGE-VL-MLLM-S2 Teacher.

## Run

```bash
cd /mnt/data/datasets/search_dataset/zxl/benchmark_build/cir_qdiag_train

CUDA_VISIBLE_DEVICES=0 \
  bash scripts/run_fashioniq_val_evaluation.sh
```

For a background run:

```bash
cd /mnt/data/datasets/search_dataset/zxl/benchmark_build/cir_qdiag_train
mkdir -p outputs/fashioniq_val

nohup env CUDA_VISIBLE_DEVICES=0 \
  bash scripts/run_fashioniq_val_evaluation.sh \
  > outputs/fashioniq_val/nohup.log 2>&1 &

echo $!
```

Feature matrices are cached after every batch. If the process is interrupted,
run the same command again to resume the incomplete matrix. Models with a final
`metrics.json` are skipped.

Useful overrides:

```bash
CLIP_IMAGE_BATCH_SIZE=128 \
TEACHER_IMAGE_BATCH_SIZE=8 \
TEACHER_QUERY_BATCH_SIZE=8 \
CUDA_VISIBLE_DEVICES=0 \
  bash scripts/run_fashioniq_val_evaluation.sh
```

Set `RUN_TEACHER=0` to run only the five CLIP models. Set `FORCE_CACHE=1` only
when intentionally rebuilding every feature cache.

## Outputs

- Final table: `outputs/fashioniq_val/ALL_MODELS_RESULTS.md`
- CSV table: `outputs/fashioniq_val/all_models_metrics.csv`
- Complete JSON: `outputs/fashioniq_val/all_models_metrics.json`
- Individual model metrics: `outputs/fashioniq_val/<model>/metrics.json`
- Resumable features: `outputs/fashioniq_val/<model>/<category>/*.npy`
