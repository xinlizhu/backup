from __future__ import annotations

import csv
import html
from pathlib import Path
from statistics import mean

from cir_qdiag.schemas import EditOperation, PinPointGroupRecord, SemanticTarget
from cir_qdiag.utils.jsonl import read_models


def export_audit_html(input_path: str | Path, output_path: str | Path) -> None:
    records = list(read_models(input_path, PinPointGroupRecord))
    cards: list[str] = []
    for record in records:
        validated_positive_ids = set(record.metadata.validated_positive_image_ids)
        reference_html = "".join(_image_figure(item, "Reference") for item in record.references)
        positive_html = "".join(
            _image_figure(
                item,
                (
                    f"Positive {index} · validated"
                    if item.image_id in validated_positive_ids
                    else f"Positive {index} · not sent"
                ),
            )
            for index, item in enumerate(record.positive_images, start=1)
        )
        instruction_html = "".join(
            f"<li><code>{html.escape(item.instruction_id)}</code>: "
            f"{html.escape(item.text)}</li>"
            for item in record.instructions
        )
        candidate_html = "".join(
            "<article class='candidate'>"
            f"<h3>{html.escape(item.negative_id)}</h3>"
            f"<p><b>Source:</b> {html.escape(item.source_text)}</p>"
            f"<p><b>Negative:</b> {html.escape(item.text)}</p>"
            f"<p><b>Cell:</b> {item.semantic_target.value} × {item.edit_operation.value} · "
            f"<b>Status:</b> {item.status}</p>"
            f"<p><b>Anchors:</b> {html.escape(item.before_span or '∅')} → "
            f"{html.escape(item.after_span or '∅')}</p>"
            f"<p><b>Review:</b> {html.escape(item.selection_reason or 'Not reviewed')}</p>"
            "</article>"
            for item in record.candidates
        )
        cards.append(
            "<section class='group'>"
            f"<h2>{html.escape(record.group_id)}</h2>"
            f"<p>{len(record.instructions)} instructions · "
            f"{len(record.positive_images)} positives · "
            f"{len(record.negatives)} accepted negatives</p>"
            f"<div class='images'>{reference_html}{positive_html}</div>"
            f"<ol>{instruction_html}</ol>{candidate_html}</section>"
        )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f"""<!doctype html><html><head><meta charset="utf-8">
<title>PinPoint-QDiag V7 Review</title><style>
body{{font-family:system-ui,sans-serif;max-width:1400px;margin:auto;padding:20px;background:#f5f5f5}}
.group,.candidate{{background:#fff;margin:16px 0;padding:16px;border:1px solid #ddd;border-radius:8px}}
.candidate{{background:#fafafa}}.images{{display:flex;gap:10px;overflow-x:auto}}
figure{{min-width:220px;margin:0}}img{{width:220px;height:220px;object-fit:contain;background:#eee}}
figcaption{{color:#666;font-size:12px}}code{{font-size:12px}}
</style></head><body><h1>PinPoint-QDiag V7: {len(records)} groups</h1>
{''.join(cards)}</body></html>""",
        encoding="utf-8",
    )


def write_quality_reports(input_path: str | Path, output_dir: str | Path) -> None:
    records = list(read_models(input_path, PinPointGroupRecord))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    generation: list[dict[str, object]] = []
    selection: list[dict[str, object]] = []
    for target in SemanticTarget:
        for operation in EditOperation:
            candidates = [
                item
                for record in records
                for item in record.candidates
                if item.semantic_target == target and item.edit_operation == operation
            ]
            accepted = [item for item in candidates if item.status == "accepted"]
            key = {
                "semantic_target": target.value,
                "edit_operation": operation.value,
            }
            generation.append(
                {
                    **key,
                    "generated_candidates": len(candidates),
                    "average_attempts": (
                        mean(item.generation_attempts for item in candidates)
                        if candidates
                        else 0.0
                    ),
                }
            )
            selection.append(
                {
                    **key,
                    "selected_candidates": len(accepted),
                    "rejected_candidates": sum(
                        item.status == "rejected" for item in candidates
                    ),
                    "failed_candidates": sum(item.status == "failed" for item in candidates),
                    "selection_rate": len(accepted) / len(candidates) if candidates else 0.0,
                }
            )
    sample_rows = [
        {
            "group_id": record.group_id,
            "num_instructions": len(record.instructions),
            "num_references": len(record.references),
            "num_positive_images": len(record.positive_images),
            "num_validated_positive_images": len(
                record.metadata.validated_positive_image_ids
            ),
            "validated_positive_image_ids": "; ".join(
                record.metadata.validated_positive_image_ids
            ),
            "generated_candidates": len(record.candidates),
            "selected_candidates": len(record.negatives),
            "operational_status": record.metadata.operational_status,
            "semantic_status": record.metadata.semantic_status,
            "errors": "; ".join(record.metadata.errors),
        }
        for record in records
    ]
    review_rows = [
        {
            "group_id": record.group_id,
            "instruction_id": item.instruction_id,
            "negative_id": item.negative_id,
            "semantic_target": item.semantic_target.value,
            "edit_operation": item.edit_operation.value,
            "source_text": item.source_text,
            "negative_text": item.text,
            "status": item.status,
            "rejection_code": item.rejection_code.value if item.rejection_code else "",
            "reason": item.selection_reason or "",
        }
        for record in records
        for item in record.candidates
    ]
    _write_csv(output / "generation_by_edit.csv", generation)
    _write_csv(output / "selection_by_edit.csv", selection)
    _write_csv(output / "group_summary.csv", sample_rows)
    _write_csv(output / "selection_reasons.csv", review_rows)


def _image_figure(item, label: str) -> str:
    return (
        f"<figure><img src='{Path(item.image_path).resolve().as_uri()}' loading='lazy'>"
        f"<figcaption>{html.escape(label)}: {html.escape(item.image_id)}</figcaption></figure>"
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
