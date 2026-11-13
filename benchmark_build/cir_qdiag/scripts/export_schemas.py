from __future__ import annotations

import json
from pathlib import Path

from cir_qdiag.schemas import (
    BatchReviewResponse,
    CIRRScreenResponse,
    GenerationBatchResponse,
    NegativeCandidate,
    ParaphraseGenerationResponse,
    ParaphraseReviewResponse,
    PositiveParaphraseRecord,
    SampleRecord,
)


SCHEMAS = {
    "cirr_screen_response.schema.json": CIRRScreenResponse,
    "generation_batch_response.schema.json": GenerationBatchResponse,
    "batch_review_response.schema.json": BatchReviewResponse,
    "negative_candidate.schema.json": NegativeCandidate,
    "final_sample.schema.json": SampleRecord,
    "paraphrase_generation_response.schema.json": ParaphraseGenerationResponse,
    "paraphrase_review_response.schema.json": ParaphraseReviewResponse,
    "positive_paraphrase_record.schema.json": PositiveParaphraseRecord,
}


def main() -> None:
    output_dir = Path(__file__).parents[1] / "schemas"
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, model in SCHEMAS.items():
        payload = json.dumps(
            model.model_json_schema(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        (output_dir / filename).write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
