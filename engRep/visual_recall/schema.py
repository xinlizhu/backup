from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


VisualType = Literal["figure", "table"]

FINAL_FIELDS = (
    "id",
    "type",
    "title",
    "image_content",
    "image_file",
    "report_title",
    "author_org",
    "publish_time",
    "original_caption",
    "source_footnote",
    "page_idx",
    "language",
    "is_blurry",
    "publish_ready",
)


@dataclass
class VisualEvidence:
    report_id: str
    visual_id: str
    visual_type: VisualType
    report_title: str
    author_org: str
    publish_time: str
    security_code: str
    page_idx: int | None
    bbox: list[float]
    original_caption: str
    original_footnote: str
    section_title: str
    context_before: list[str]
    context_after: list[str]
    nearby_page_text: list[str]
    neighboring_visuals: list[dict[str, Any]]
    table_body: str
    target_image: Path | None
    page_image: Path | None
    mapping_warnings: list[str] = field(default_factory=list)
    asset_info: dict[str, Any] = field(default_factory=dict)

    @property
    def global_id(self) -> str:
        return f"{self.report_id}:{self.visual_type}:{self.visual_id}"

    def prompt_payload(self, max_table_chars: int = 40_000) -> dict[str, Any]:
        table_body = self.table_body
        truncated = False
        if len(table_body) > max_table_chars:
            half = max_table_chars // 2
            table_body = table_body[:half] + "\n...[TRUNCATED]...\n" + table_body[-half:]
            truncated = True
        return {
            "report_metadata": {
                "report_id": self.report_id,
                "report_title": self.report_title,
                "author_org": self.author_org,
                "publish_time": self.publish_time,
                "security_code": self.security_code,
            },
            "visual": {
                "visual_id": self.visual_id,
                "type": self.visual_type,
                "page_idx_zero_based": self.page_idx,
                "bbox_normalized": self.bbox,
                "original_caption": self.original_caption,
                "section_title": self.section_title,
            },
            "context_before": self.context_before,
            "context_after": self.context_after,
            "nearby_page_text": self.nearby_page_text,
            "neighboring_visuals_on_same_page": self.neighboring_visuals,
            "table_body": table_body if self.visual_type == "table" else "",
            "table_body_truncated": truncated,
            "mapping_warnings": self.mapping_warnings,
        }


def build_final_record(
    evidence: VisualEvidence,
    *,
    title: str,
    image_content: str,
    image_file: Path,
    source_footnote: str,
    language: str,
    is_blurry: bool,
    publish_ready: bool,
) -> dict[str, Any]:
    record = {
        "id": evidence.global_id,
        "type": evidence.visual_type,
        "title": title.strip(),
        "image_content": image_content.strip(),
        "image_file": str(image_file.resolve()),
        "report_title": evidence.report_title.strip(),
        "author_org": evidence.author_org.strip(),
        "publish_time": evidence.publish_time.strip(),
        "original_caption": evidence.original_caption.strip(),
        "source_footnote": source_footnote.strip(),
        "page_idx": evidence.page_idx,
        "language": language,
        "is_blurry": bool(is_blurry),
        "publish_ready": bool(publish_ready),
    }
    assert tuple(record) == FINAL_FIELDS
    return record


def validate_final_record(record: dict[str, Any], *, strict_file: bool = True, allow_unknown_figure_page: bool = False) -> list[str]:
    errors: list[str] = []
    if tuple(record) != FINAL_FIELDS:
        errors.append("final field order/schema mismatch")
    if record.get("type") not in {"figure", "table"}:
        errors.append("type must be figure or table")
    for key in ("id", "title", "image_content", "image_file", "report_title", "author_org", "publish_time"):
        if not isinstance(record.get(key), str) or not record[key].strip():
            errors.append(f"{key} is empty")
    unknown_figure_page = allow_unknown_figure_page and record.get("type") == "figure" and record.get("page_idx") is None
    if not unknown_figure_page and (not isinstance(record.get("page_idx"), int) or record["page_idx"] < 0):
        errors.append("page_idx must be a non-negative integer")
    if not isinstance(record.get("is_blurry"), bool):
        errors.append("is_blurry must be boolean")
    if not isinstance(record.get("publish_ready"), bool):
        errors.append("publish_ready must be boolean")
    if strict_file and record.get("image_file") and not Path(record["image_file"]).is_file():
        errors.append("image_file does not exist")
    return errors
