#!/usr/bin/env python3
"""Combine per-model FashionIQ validation metrics into CSV and Markdown."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from cir_qdiag_train.fashioniq_evaluation import FASHIONIQ_CATEGORIES
from cir_qdiag_train.on_policy_mining import write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        help="MODEL_NAME=/path/to/metrics.json (repeat for each model)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    payloads = []
    for specification in args.result:
        if "=" not in specification:
            raise ValueError(f"invalid --result {specification!r}; expected MODEL=PATH")
        model_name, raw_path = specification.split("=", 1)
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("model") != model_name:
            raise ValueError(
                f"result label mismatch for {path}: expected {model_name!r}, "
                f"found {payload.get('model')!r}"
            )
        payloads.append(payload)
        row = {"Model": model_name}
        for category in FASHIONIQ_CATEGORIES:
            metrics = payload["categories"][category]
            row[f"{category.title()} R@10"] = metrics["recall_at_10"]
            row[f"{category.title()} R@50"] = metrics["recall_at_50"]
        row["Average R@10"] = payload["macro_average"]["recall_at_10"]
        row["Average R@50"] = payload["macro_average"]["recall_at_50"]
        rows.append(row)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_dir / "all_models_metrics.json", payloads)
    columns = list(rows[0])
    with (output_dir / "all_models_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# FashionIQ validation results",
        "",
        "Protocol: concatenate the two provided captions in stored order with `and`; "
        "retrieve within each category-specific validation gallery; keep the reference "
        "image in the gallery. Average is the macro average over dress, shirt, and toptee.",
        "",
        "FashionIQ's standard headline metrics are R@10 and R@50; all per-model JSON "
        "files additionally contain R@1, R@5, and a query-weighted micro average.",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" if column == "Model" else "---:" for column in columns) + "|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                str(row[column]) if column == "Model" else f"{float(row[column]):.4f}"
                for column in columns
            )
            + " |"
        )
    markdown = "\n".join(lines) + "\n"
    (output_dir / "ALL_MODELS_RESULTS.md").write_text(markdown, encoding="utf-8")
    print(markdown, flush=True)


if __name__ == "__main__":
    main()
