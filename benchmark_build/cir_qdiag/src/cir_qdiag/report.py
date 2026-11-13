from __future__ import annotations

import csv
import html
from pathlib import Path
from statistics import mean

from cir_qdiag.schemas import (
    ConstructionMode,
    EditOperation,
    SampleRecord,
    SemanticTarget,
)
from cir_qdiag.utils.jsonl import read_models


def export_audit_html(input_path: str | Path, output_path: str | Path) -> None:
    """Export the generated text and the one final selection decision."""

    records = list(read_models(input_path, SampleRecord))
    cards: list[str] = []
    for record in records:
        if not record.candidates:
            cards.append(
                f'<section class="card"><h2>{html.escape(record.sample_id)}</h2>'
                f"<p><b>Positive:</b> {html.escape(record.positive_text)}</p>"
                '<p class="muted">No generated candidate.</p></section>'
            )
            continue
        for candidate in record.candidates:
            label = (
                f"{candidate.semantic_target.value} × {candidate.edit_operation.value} × "
                f"{candidate.construction_mode.value}"
            )
            basis = candidate.visual_basis or "—"
            before = candidate.before_span or "∅"
            after = candidate.after_span or "∅"
            reason = candidate.selection_reason or "Not reviewed"
            cards.append(
                f'<section class="card" data-negative-id="{html.escape(candidate.negative_id)}">'
                f"<h2>{html.escape(candidate.negative_id)}</h2>"
                '<div class="images">'
                f'<figure><img src="{Path(record.reference.image_path).resolve().as_uri()}" loading="lazy">'
                f"<figcaption>Reference: {html.escape(record.reference.image_id)}</figcaption></figure>"
                f'<figure><img src="{Path(record.target.image_path).resolve().as_uri()}" loading="lazy">'
                f"<figcaption>Target: {html.escape(record.target.image_id)}</figcaption></figure>"
                "</div>"
                f"<p><b>Positive:</b> {html.escape(record.positive_text)}</p>"
                f"<p><b>Candidate:</b> {html.escape(candidate.text)}</p>"
                f"<p><b>Cell:</b> {html.escape(label)} · "
                f"<b>Status:</b> {html.escape(candidate.status)}</p>"
                f"<p><b>Generated edit:</b> {html.escape(candidate.change_summary)}</p>"
                f"<p><b>Edit anchors:</b> {html.escape(before)} → {html.escape(after)}</p>"
                f"<p><b>Visual basis:</b> {html.escape(basis)}</p>"
                f"<p><b>Final review:</b> {html.escape(reason)}</p>"
                "</section>"
            )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>CIR-QDiag V5 Review</title>
<style>
body {{ font-family: system-ui,sans-serif; max-width:1200px; margin:auto; padding:20px; background:#f5f5f5; }}
.toolbar {{ position:sticky; top:0; background:#fff; padding:12px; border:1px solid #ddd; z-index:2; }}
.card {{ background:#fff; margin:18px 0; padding:18px; border:1px solid #ddd; border-radius:8px; }}
.images {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
figure {{ margin:0; }} img {{ width:100%; height:360px; object-fit:contain; background:#eee; }}
figcaption,.muted {{ color:#666; }}
</style></head><body>
<div class="toolbar"><b>CIR-QDiag V5 automatic review: {len(records)} samples</b></div>
{"".join(cards)}
</body></html>""",
        encoding="utf-8",
    )


def write_quality_reports(input_path: str | Path, output_dir: str | Path) -> None:
    records = list(read_models(input_path, SampleRecord))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    generation: list[dict[str, object]] = []
    selection: list[dict[str, object]] = []
    final: list[dict[str, object]] = []

    for target in SemanticTarget:
        for operation in EditOperation:
            for mode in ConstructionMode:
                candidates = [
                    item
                    for record in records
                    for item in record.candidates
                    if item.semantic_target == target
                    and item.edit_operation == operation
                    and item.construction_mode == mode
                ]
                accepted = [item for item in candidates if item.status == "accepted"]
                key = {
                    "semantic_target": target.value,
                    "edit_operation": operation.value,
                    "construction_mode": mode.value,
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
                        "failed_candidates": sum(
                            item.status == "failed" for item in candidates
                        ),
                        "selection_rate": (
                            len(accepted) / len(candidates) if candidates else 0.0
                        ),
                    }
                )
                final.append({**key, "final_negative_count": len(accepted)})

    accepted_counts = [len(record.negatives) for record in records]
    final.append(
        {
            "semantic_target": "ALL",
            "edit_operation": "ALL",
            "construction_mode": "ALL",
            "final_negative_count": sum(accepted_counts),
            "average_negatives_per_triplet": (
                mean(accepted_counts) if accepted_counts else 0.0
            ),
            "triplets_without_valid_negative": sum(count == 0 for count in accepted_counts),
        }
    )
    sample_rows = [
        {
            "sample_id": record.sample_id,
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
            "sample_id": record.sample_id,
            "negative_id": item.negative_id,
            "semantic_target": item.semantic_target.value,
            "edit_operation": item.edit_operation.value,
            "construction_mode": item.construction_mode.value,
            "before_span": item.before_span or "",
            "after_span": item.after_span or "",
            "status": item.status,
            "reason": item.selection_reason or "",
        }
        for record in records
        for item in record.candidates
    ]
    _write_csv(output / "generation_by_edit.csv", generation)
    _write_csv(output / "selection_by_edit.csv", selection)
    _write_csv(output / "final_distribution.csv", final)
    _write_csv(output / "sample_summary.csv", sample_rows)
    _write_csv(output / "selection_reasons.csv", review_rows)


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
