from __future__ import annotations

import json
from pathlib import Path

import _bootstrap  # noqa: F401

from cir_qdiag.schemas import (
    GenerationBatchResponse,
    GroupReviewResponse,
    NegativeCandidate,
    PinPointGroupRecord,
)


SCHEMAS = {
    "generation_batch_response.schema.json": GenerationBatchResponse,
    "group_review_response.schema.json": GroupReviewResponse,
    "negative_candidate.schema.json": NegativeCandidate,
    "final_group.schema.json": PinPointGroupRecord,
}


def main() -> None:
    output_dir = Path(__file__).parents[1] / "schemas"
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob("*.json"):
        path.unlink()
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
