# PinPoint-QDiag V7

Generate structured hard-negative **texts** from exact PinPoint paraphrase groups with an
OpenAI-compatible Chat Completions API.

## Dataset unit

`prepare` reads `pinpoint_licensed.parquet`, removes `None` values and duplicates from candidate
arrays, and groups rows by the exact tuple:

```text
(reference image 1, optional reference image 2,
 exact positive-image set, exact official-negative-image set)
```

Only groups containing 2–6 instructions are retained. On the current licensed release this gives:

- 1,683 groups
- 6,553 instructions
- 987 singleton instructions excluded
- 9 oversized groups (7–17 instructions) excluded

There is no model-based source screening.

## Generation and validation

Every instruction independently receives three **text-only** generation calls:

1. ADD → object, attribute, predicate
2. DELETE → object, attribute, predicate
3. SUBSTITUTE → object, attribute, predicate

Each instruction therefore has up to nine candidates. No image is sent during generation.
All non-null proposals are passed to the group-level model review. Generation does not use local
regex or semantic gates to silently discard proposed edits; `before_span` and `after_span` are
treated as untrusted generator claims during review.

Every group then receives exactly one validation call containing:

1. one or two Reference images;
2. at most eight Positive images selected deterministically from the full set;
3. every generated negative text from every instruction in the group.

PinPoint's official negative images are retained as IDs in output metadata but are never sent to
generation or validation. The full Positive set remains in each group record, while
`metadata.validated_positive_image_ids` records the exact subset sent for validation.

ADD and SUBSTITUTE are accepted only when their new fact is false for every supplied Positive
image. DELETE is accepted when it removes exactly one meaningful constraint and produces a fluent,
strictly weaker query; it is not required to contradict the Positive images.

## Setup

```bash
cd /mnt/data/datasets/search_dataset/zxl/benchmark_build/cir_qdiag_openai
python -m pip install -e '.[dev]'

export OPENAI_BASE_URL='https://your-host.example/v1'
export OPENAI_API_KEY='YOUR_KEY'
```

The endpoint must support:

```text
POST /v1/chat/completions
multimodal image_url content
```

Requests are sent with a minimal `httpx` client rather than the OpenAI Python
SDK because some compatible gateways reject the SDK's additional Stainless
headers. The client uses the same compact headers as a working curl request and
disables environment-proxy inheritance.

By default, the JSON Schema is included in the system prompt and every response
is strictly validated locally with Pydantic. Set
`model.structured_output_mode: json_schema` only when the provider reliably
supports strict structured outputs.

## Run

Prepare groups locally without API calls:

```bash
python scripts/prepare_pinpoint.py \
  --config configs/pilot_50.yaml \
  --output outputs/pinpoint_pilot_50_v7/groups.jsonl
```

Run the complete 50-group pilot:

```bash
python scripts/run_pipeline.py --config configs/pilot_50.yaml
```

Export all eligible exact groups without making API calls:

```bash
python scripts/prepare_pinpoint.py \
  --config configs/full.yaml \
  --output outputs/pinpoint_full_v7/groups.jsonl
```

After checking the full call count below, the complete run is:

```bash
python scripts/run_pipeline.py --config configs/full.yaml
```

Run stages independently:

```bash
python scripts/generate_negatives.py \
  --config configs/pilot_50.yaml \
  --input outputs/pinpoint_pilot_50_v7/groups.jsonl \
  --output outputs/pinpoint_pilot_50_v7/candidates.jsonl

python scripts/validate_negatives.py \
  --config configs/pilot_50.yaml \
  --input outputs/pinpoint_pilot_50_v7/candidates.jsonl \
  --output outputs/pinpoint_pilot_50_v7/final.jsonl
```

Use `--force` only to rebuild that stage. Normal runs resume from completed group IDs and reuse
request caches.

Provider-wide account failures such as `DAILY_LIMIT_EXCEEDED` stop the active stage immediately
instead of recording every remaining group as a failed sample. After restoring quota, rerun the
same command with `--resume`; completed groups and cached calls are retained, while rows whose
`metadata.operational_status` is `failed` are regenerated and overwritten.

## Call volume

For `G` groups with `I` total instructions:

```text
generation calls = 3 × I
validation calls = G
total calls = 3 × I + G
```

The current full retained dataset would require:

```text
3 × 6,553 + 1,683 = 21,342 API calls
```

The deterministic 50-group pilot at seed 42 contains 196 instructions:

```text
588 generation calls + 50 validation calls = 638 API calls
```

## Outputs

```text
groups.jsonl       exact 2–6-instruction source groups
candidates.jsonl   generated 3×3 candidates
final.jsonl        candidates plus one group-level validation verdict each
audit.html         references, positives, instructions, candidates, verdicts
generation_by_edit.csv
selection_by_edit.csv
group_summary.csv
selection_reasons.csv
cache/             structured API response cache
```

The V7 group schema is exported in `schemas/final_group.schema.json`.

## Tests

```bash
pytest -q
ruff check src tests scripts
```
