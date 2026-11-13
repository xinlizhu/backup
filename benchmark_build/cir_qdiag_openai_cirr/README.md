# CIRR QDiag V8

Use an OpenAI-compatible multimodal Chat Completions API to build hard-negative modification
instructions for the official CIRR training split. Generation is text-only; final review sees the
Reference and Target images.

## What the pipeline does

`prepare` reads:

- `captions/cap.rc2.train.json` (28,225 train triplets);
- `image_splits/split.rc2.train.json`;
- the CIRR image root.

Each output record is one official CIRR training triplet:

```text
(sample_id, pair_id, reference image metadata, target image metadata,
 positive_text, image_set)
```

The image paths are retained for downstream BGE-VL training and the local HTML audit. Generation
never sends them to GPT. Final validation sends the Reference image followed by the Target image.

For every `positive_text`, generation makes three text-only calls:

1. ADD → independently propose object, attribute, predicate;
2. DELETE → independently propose object, attribute, predicate;
3. SUBSTITUTE → independently propose object, attribute, predicate.

Each cell may be `null`, so a sample has 0–9 candidates. One final multimodal call reviews all
non-null candidates for single-edit correctness, semantic axis, operation, fluency, closeness, and
visual counterfactual validity. Accepted candidates are copied into `negatives`, with:

```json
{
  "semantic_target": "object | attribute | predicate",
  "edit_operation": "add | delete | substitute",
  "construction_mode": "text_only_editable",
  "status": "accepted"
}
```

This format can be consumed directly by
`/mnt/data/datasets/search_dataset/zxl/benchmark_build/cir_qdiag_train`.

### Generation and validation boundary

Generation is strictly text-only and cannot introduce facts from the images. Validation receives
Image 1 as Reference and Image 2 as the Target positive. Reference resolves explicit relative
phrases; ADD and SUBSTITUTE are accepted only when their new fact is clearly false in Target.
DELETE remains a weaker-query diagnostic and need not contradict Target. Every record carries
`metadata.validation_mode="reference_target_visual"`.

## Setup

```bash
cd /mnt/data/datasets/search_dataset/zxl/benchmark_build/cir_qdiag_openai_cirr
python -m pip install -e '.[dev]'

export OPENAI_BASE_URL='https://your-host.example/v1'
export OPENAI_API_KEY='YOUR_KEY'
```

This project installs the independent Python package `cir_qdiag_openai_cirr`; it does not replace
the original Qwen project's `cir_qdiag` package, so both editable installs can coexist.

The code sends:

```text
POST ${OPENAI_BASE_URL}/chat/completions
```

The endpoint must support multimodal `image_url` content in Chat Completions. Images are embedded as
data URLs, with `model.image_detail` controlling detail. JSON Schema is included in the system
prompt by default and responses are strictly validated by Pydantic. Set
`model.structured_output_mode: json_schema` only if the gateway supports strict structured output.

## Run

First verify the real train loader without using the API:

```bash
python scripts/prepare_cirr.py \
  --config configs/pilot_50.yaml \
  --output outputs/cirr_train_pilot_50_v8/source.jsonl
```

Test the text endpoint:

```bash
python scripts/smoke_test_openai.py --config configs/pilot_50.yaml
```

Run a 50-sample pilot:

```bash
python scripts/run_pipeline.py --config configs/pilot_50.yaml
```

Run the full train split:

```bash
python scripts/run_pipeline.py --config configs/full.yaml
```

### Generate equivalent positive paraphrases

This independent text-only stage reads an existing CIRR V8 JSONL file. For every `positive_text`,
one OpenAI call generates the requested variants and a second OpenAI call reviews complete semantic
equivalence. Reference and Target images are retained in the output but are not sent.

For the completed 500-sample pilot:

```bash
python scripts/generate_positive_paraphrases.py \
  --config configs/pilot_500.yaml \
  --input outputs/cirr_train_pilot_500_v8/final.jsonl \
  --output outputs/cirr_train_pilot_500_v8/positive_paraphrases.jsonl \
  --num-variants 5
```

The command resumes by `sample_id`, retries failed rows, and reuses cached structured responses.
Use `--force` only when deliberately rebuilding every paraphrase row.

Stages can also be run independently:

```bash
python scripts/generate_negatives.py \
  --config configs/pilot_50.yaml \
  --input outputs/cirr_train_pilot_50_v8/source.jsonl \
  --output outputs/cirr_train_pilot_50_v8/candidates.jsonl

python scripts/validate_negatives.py \
  --config configs/pilot_50.yaml \
  --input outputs/cirr_train_pilot_50_v8/candidates.jsonl \
  --output outputs/cirr_train_pilot_50_v8/final.jsonl
```

Normal runs resume by `sample_id` and reuse cached structured responses. Use `--force-stage` only
when deliberately rebuilding a stage. Provider-wide quota/account failures abort the active stage
instead of writing thousands of failed rows; rerun after restoring quota to resume.

For a shard of the selected records:

```bash
python scripts/run_pipeline.py \
  --config configs/full.yaml \
  --start-index 0 \
  --end-index 1000
```

Use separate `runtime.output_dir` values for concurrently executed shards.

## Call volume

For `N` CIRR samples, of which `M` have at least one non-null candidate:

```text
generation calls = 3 × N
validation calls = M
total calls = 3 × N + M
```

The full 28,225-sample train split therefore requires at most 112,900 API calls. A 50-sample pilot
requires at most 200 calls. Null generation slots do not change the three generation calls, but a
sample with all nine slots null skips validation.

## Outputs

```text
source.jsonl          CIRR train source records
candidates.jsonl      up to nine generated candidates per sample
final.jsonl           candidates plus visual verdicts and accepted negatives
audit.html            local reference/target preview and all text verdicts
generation_by_edit.csv
selection_by_edit.csv
sample_summary.csv
selection_reasons.csv
positive_paraphrases.jsonl
positive_paraphrase_metrics.csv
positive_paraphrase_summary.json
cache/                structured API response cache
```

Exported JSON Schemas are under `schemas/`, including `final_sample.schema.json`.

## Tests

```bash
pytest -q
ruff check src tests scripts
```
