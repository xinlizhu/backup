from pathlib import Path

from cir_qdiag.report import export_audit_html, write_quality_reports
from cir_qdiag.schemas import ImageRef, NegativeCandidate, SampleRecord
from cir_qdiag.utils.jsonl import write_jsonl_atomic


def test_simplified_record_exports_html_and_reports(tmp_path: Path) -> None:
    candidate = NegativeCandidate(
        negative_id="s_attribute_substitute_text",
        text="Show a blue car",
        semantic_target="attribute",
        edit_operation="substitute",
        construction_mode="text_only_editable",
        before_span="red",
        after_span="blue",
        change_summary="red to blue",
        status="accepted",
        selection_reason="Blue contradicts the red target car.",
    )
    record = SampleRecord(
        sample_id="s",
        pair_id=1,
        reference=ImageRef(image_id="r", image_path=str(tmp_path / "r.png")),
        target=ImageRef(image_id="t", image_path=str(tmp_path / "t.png")),
        positive_text="Show a red car",
        candidates=[candidate],
        negatives=[candidate.model_copy(deep=True)],
        metadata={
            "stage": "final",
            "operational_status": "ok",
            "semantic_status": "accepted",
            "num_generated_candidates": 1,
            "num_accepted_negatives": 1,
        },
    )
    source = tmp_path / "final.jsonl"
    write_jsonl_atomic(source, [record])
    export_audit_html(source, tmp_path / "audit.html")
    write_quality_reports(source, tmp_path / "reports")
    assert "Blue contradicts" in (tmp_path / "audit.html").read_text(encoding="utf-8")
    assert "red → blue" in (tmp_path / "audit.html").read_text(encoding="utf-8")
    assert (tmp_path / "reports" / "selection_by_edit.csv").is_file()
    assert (tmp_path / "reports" / "selection_reasons.csv").is_file()
