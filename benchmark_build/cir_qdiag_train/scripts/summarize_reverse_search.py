from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

FIELDS = (
    "num_reverse_targets",
    "num_query_candidates",
    "target_hit_rate_at_1",
    "target_hit_rate_at_5",
    "target_hit_rate_at_10",
    "target_hit_rate_at_50",
    "macro_query_recall_at_10",
    "pair_recall_at_10",
    "map_at_10",
    "ndcg_at_10",
    "mrr",
    "mean_best_positive_rank",
    "median_best_positive_rank",
    "mean_positive_rank",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize reverse CIRR retrieval runs.")
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="MODEL=JSON",
        help="Model label and reverse metric JSON; repeat for each model.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for specification in args.input:
        if "=" not in specification:
            raise ValueError(f"invalid --input {specification!r}; expected MODEL=JSON")
        model, raw_path = specification.split("=", 1)
        path = Path(raw_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        missing = [field for field in FIELDS if field not in payload]
        if missing:
            raise ValueError(f"{path} is missing reverse metrics: {missing}")
        rows.append(
            {
                "model": model,
                "checkpoint": payload["checkpoint"],
                **{field: payload[field] for field in FIELDS},
            }
        )

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)
    print(output.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
