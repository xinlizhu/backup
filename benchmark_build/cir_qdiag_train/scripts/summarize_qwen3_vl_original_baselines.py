#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


MODELS = (
    "Qwen3-VL-Embedding-2B",
    "Qwen3-VL-Embedding-8B",
)
QDIAG_SUBSETS = (
    "strict",
    "add",
    "substitute",
    "delete",
    "strict|object",
    "strict|attribute",
    "strict|predicate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.output_root.resolve()
    records = [_load_model(root, model) for model in MODELS]
    _write_combined_qdiag(root, records)
    _write_combined_pinpoint(root, records)
    _write_json_atomic(root / "all_metrics.json", {"models": records})
    _write_text_atomic(
        root / "QWEN3_VL_ORIGINAL_RESULTS.md",
        "\n".join(_markdown(records)).rstrip() + "\n",
    )


def _load_model(root: Path, model: str) -> dict[str, Any]:
    cirr = _read_json(root / "cirr" / model / "val_metrics.json")
    fashioniq = _read_json(root / "fashioniq" / model / "metrics.json")
    pinpoint = _read_single_csv(
        root / "pinpoint_official" / model / "official_metrics.csv"
    )
    qdiag_rows = _read_csv(root / "qdiag" / model / "summary.csv")
    qdiag = {row["subset"]: row for row in qdiag_rows}
    return {
        "model": model,
        "cirr": cirr,
        "fashioniq": fashioniq,
        "pinpoint": pinpoint,
        "qdiag": qdiag,
    }


def _markdown(records: list[dict[str, Any]]) -> list[str]:
    lines = [
        "# Original Qwen3-VL-Embedding baselines",
        "",
        "两个模型均使用官方 pooled embedding：最后一个非 padding token pooling 后做 "
        "L2 normalization。组合检索 query 是 `reference image(s) + modification text`，"
        "候选是 image-only；二者使用不同的 retrieval instruction。所有结果均为零样本，"
        "没有在 CIRR、PinPoint、FashionIQ 或 CIR-QDiag 上训练。图像预处理固定为 "
        "`min_pixels=4×32²`、`max_pixels=256×32²`，与现有 CIR-QDiag 的 Qwen 协议一致。",
        "",
        "## Completion",
        "",
        "| Model | CIRR | QDiag | FashionIQ | PinPoint full |",
        "|---|---:|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            f"| {record['model']} | {_done(record['cirr'])} | "
            f"{_done(record['qdiag'])} | {_done(record['fashioniq'])} | "
            f"{_done(record['pinpoint'])} |"
        )

    lines.extend(
        [
            "",
            "## CIRR val",
            "",
            "| Model | R@1 | R@5 | R@10 | R@50 | Group R@1 | Group R@2 | Group R@3 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for record in records:
        row = record["cirr"]
        lines.append(
            f"| {record['model']} | {_metric(row, 'recall_at_1')} | "
            f"{_metric(row, 'recall_at_5')} | {_metric(row, 'recall_at_10')} | "
            f"{_metric(row, 'recall_at_50')} | "
            f"{_metric(row, 'group_recall_at_1')} | "
            f"{_metric(row, 'group_recall_at_2')} | "
            f"{_metric(row, 'group_recall_at_3')} |"
        )

    lines.extend(
        [
            "",
            "## Official PinPoint",
            "",
            "| Model | mAP@10 ↑ | NegRecall@10 ↓ | Queries |",
            "|---|---:|---:|---:|",
        ]
    )
    for record in records:
        row = record["pinpoint"]
        lines.append(
            f"| {record['model']} | {_metric(row, 'mAP@10')} | "
            f"{_metric(row, 'NegRecall@10')} | {_integer(row, 'total_queries')} |"
        )

    lines.extend(
        [
            "",
            "## CIR-QDiag strict ADD/SUBSTITUTE",
            "",
            "| Model | tau95 | Mean Delta | Raw Win | Stable Win | Own target - random target |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for record in records:
        row = record["qdiag"].get("strict", {})
        lines.append(
            f"| {record['model']} | "
            f"{_metric(row, 'normal_perturbation_threshold', 6)} | "
            f"{_metric(row, 'mean_target_delta', 6)} | "
            f"{_metric(row, 'target_raw_win_rate', 6)} | "
            f"{_metric(row, 'target_stable_win_rate', 6)} | "
            f"{_metric(row, 'positive_query_own_minus_random_target', 6)} |"
        )

    lines.extend(
        [
            "",
            "### Fine-grained Stable Win",
            "",
            "| Model | ADD | SUBSTITUTE | DELETE | Object | Attribute | Predicate |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for record in records:
        rows = record["qdiag"]
        values = [
            _metric(rows.get(subset, {}), "target_stable_win_rate", 6)
            for subset in QDIAG_SUBSETS[1:]
        ]
        lines.append(f"| {record['model']} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## FashionIQ val",
            "",
            "| Model | Average R@1 | Average R@5 | Average R@10 | Average R@50 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for record in records:
        row = record["fashioniq"].get("macro_average", {})
        lines.append(
            f"| {record['model']} | {_metric(row, 'recall_at_1')} | "
            f"{_metric(row, 'recall_at_5')} | {_metric(row, 'recall_at_10')} | "
            f"{_metric(row, 'recall_at_50')} |"
        )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- Complete structured metrics: `all_metrics.json`",
            "- Combined QDiag rows: `qdiag/combined_summary_all_models.csv`",
            "- Combined PinPoint headline metrics: "
            "`pinpoint_official/official_map10_negrecall10_all_models.csv`",
        ]
    )
    return lines


def _write_combined_qdiag(root: Path, records: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for record in records:
        for subset, values in record["qdiag"].items():
            rows.append({"model": record["model"], "subset": subset, **values})
    if rows:
        _write_csv_atomic(root / "qdiag" / "combined_summary_all_models.csv", rows)


def _write_combined_pinpoint(root: Path, records: list[dict[str, Any]]) -> None:
    rows = [
        {
            "model": record["model"],
            "mAP@10": record["pinpoint"]["mAP@10"],
            "NegRecall@10": record["pinpoint"]["NegRecall@10"],
            "total_queries": record["pinpoint"]["total_queries"],
        }
        for record in records
        if record["pinpoint"]
    ]
    if rows:
        _write_csv_atomic(
            root
            / "pinpoint_official"
            / "official_map10_negrecall10_all_models.csv",
            rows,
        )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_single_csv(path: Path) -> dict[str, str]:
    rows = _read_csv(path)
    if not rows:
        return {}
    if len(rows) != 1:
        raise ValueError(f"Expected one row in {path}, found {len(rows)}")
    return rows[0]


def _done(value: Any) -> str:
    return "yes" if value else "pending"


def _metric(row: dict[str, Any], key: str, digits: int = 4) -> str:
    value = row.get(key)
    if value in (None, ""):
        return "—"
    return f"{float(value):.{digits}f}"


def _integer(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if value in (None, ""):
        return "—"
    return str(int(float(value)))


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    main()
