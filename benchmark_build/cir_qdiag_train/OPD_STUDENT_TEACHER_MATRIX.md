# Rank-OPD Student–Teacher matrix

## Experiment cells

The controlled matrix uses the original CIRR train set, Student-selected dynamic
Top-32 candidates, a forced ground-truth target, and the same Teacher target
Top-10 validity rule.

| Student | Teacher | Training form | Status |
| --- | --- | --- | --- |
| CLIP-L | BGE-VL-MLLM-S2 | Full finetune | Existing checkpoint; not retrained |
| CLIP-B | BGE-VL-MLLM-S2 | Full finetune | New |
| CLIP-B | Qwen3-VL-Embedding-8B | Full finetune | New |
| CLIP-L | Qwen3-VL-Embedding-8B | Full finetune | New |
| Qwen3-VL-Embedding-2B | Qwen3-VL-Embedding-8B | Query-side LoRA, fixed image/index encoder | New |

All new runs use 4 epochs and save after every epoch. CLIP uses a
physical batch size of 32. Qwen-2B uses physical batch 8 and gradient
accumulation 4, so its effective query batch is also 32.

Qwen-2B is deliberately asymmetric. Its original image encoder is used with
LoRA disabled and its gallery is cached once. The composed-query encoder uses
language-attention LoRA. The current query policy re-mines Top-K every epoch.
This is an exact on-policy candidate set for that deployed asymmetric model and
does not require backpropagating through roughly 1,000 Qwen image encodings per
optimizer step.

## Loss and its relationship to MiniLLM

The existing CLIP-L ← S2 result used forward listwise KL, so the controlled
matrix keeps the same objective:

\[
\mathcal L_{\mathrm{rank}}
= D_{\mathrm{KL}}\left(p_T(\cdot\mid x,C_\theta(x))
\;\|\;p_\theta(\cdot\mid x,C_\theta(x))\right).
\]

The support `C_theta(x)` is on-policy because it is the current Student's
Top-K. This is related to MiniLLM through Student-policy sampling, but it is not
MiniLLM's reverse-KL objective. The implementation now exposes
`objective.rank_kl_direction: forward|reverse`; reverse mode computes the exact
categorical Top-K objective

\[
D_{\mathrm{KL}}(p_\theta\|p_T),
\]

without a policy-gradient estimator because all 32 retrieval outcomes and both
distributions are explicitly available.

## Run

```bash
cd /mnt/data/datasets/search_dataset/zxl/benchmark_build/cir_qdiag_train

mkdir -p outputs/opd_student_teacher_matrix
nohup env CUDA_VISIBLE_DEVICES=0 \
  bash scripts/run_opd_student_teacher_matrix.sh \
  > outputs/opd_student_teacher_matrix/nohup.log 2>&1 &

echo $!
```

The runner is resumable. It reuses the existing S2 Teacher cache, builds the
Qwen-8B Teacher cache once, builds the immutable Qwen-2B gallery once, resumes
from each run's `last.pt`, and skips completed final checkpoints/evaluations.
The existing 109,598-image original Qwen-2B PinPoint corpus cache is reused for
the trained Qwen-2B Student because its image/index encoder is identical.

Useful overrides include:

```bash
QWEN8_TEACHER_IMAGE_BATCH_SIZE=4
QWEN8_TEACHER_QUERY_BATCH_SIZE=4
QWEN2_GALLERY_BATCH_SIZE=16
QWEN2_EVAL_BATCH_SIZE=16
CLIP_EVAL_BATCH_SIZE=128
RUN_EXTENDED_EVAL=0
```

`RUN_EXTENDED_EVAL=0` runs cache preparation, training, and CIRR validation but
skips PinPoint, CIR-QDiag, and FashionIQ. Re-running later with the default
value completes those evaluations.

Final results are collected in
`outputs/opd_student_teacher_matrix/RESULTS.md`; raw values are also written to
`all_metrics.csv`. Per-epoch timing is recorded in each run's
`training_time.json`.

## Additional experiments required for a MiniLLM-motivated claim

1. **Forward-KL versus reverse-KL.** Run at least CLIP-L ← S2 and Qwen-2B ←
   Qwen-8B with both directions. Without this, MiniLLM is motivation for
   on-policy sampling, not evidence for its central reverse-KL result.
2. **Matched valid-query comparison.** S2 and Qwen-8B will retain different
   numbers of Teacher-Top-10 queries. Compare them again on the intersection of
   queries valid for both Teachers, or match the retained sample count.
3. **Entropy/temperature control.** A common temperature of 0.05 can produce
   different Teacher distribution entropies. The training log now records
   `teacher_rank_entropy`; add an entropy-matched temperature comparison if the
   two Teachers differ materially.
4. **Complete the balanced Teacher control.** Qwen-2B ← S2 is the most useful
   missing cell. It separates the effect of Qwen-8B as Teacher from the effect
   of using a Qwen Student.
5. **Optimization-form control.** CLIP is full-finetuned while Qwen-2B uses
   query LoRA. A CLIP query-LoRA control, or a smaller Qwen full/last-N run,
   determines how much of the result comes from the adaptation form.
6. **Three seeds for the final two headline cells.** Use seeds 42/43/44 after
   choosing the KL direction and Teacher-temperature protocol; report mean and
   standard deviation on CIRR R@1, FashionIQ R@10, PinPoint mAP@10 and QDiag
   Stable Win.
