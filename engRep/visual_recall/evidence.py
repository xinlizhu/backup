from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageOps

from .schema import VisualEvidence, VisualType


SOURCE_RE = re.compile(r"\bsource\s*:|资料来源|\bas\s+of\b|截至", re.IGNORECASE)
LOW_VALUE_PAGE_TEXT_RE = re.compile(
    r"@|\bplease see page\b|\bimportant disclosures\b|\brequired analyst certifications\b|"
    r"\bdoes and seeks to do business\b",
    re.IGNORECASE,
)
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff")


@dataclass
class RefInfo:
    visual_id: str
    visual_type: VisualType
    sequence_name: str
    position: int
    original_context: str
    section_title: str
    context_before: list[str]
    context_after: list[str]


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(_clean_text(item) for item in value if _clean_text(item)).strip()
    return re.sub(r"\s+", " ", str(value)).strip()


def _block_text(block: Any) -> str:
    if not isinstance(block, dict):
        return ""
    for key in ("text", "context"):
        text = _clean_text(block.get(key))
        if text:
            return text
    return ""


def _nearby_text(sequence: list[Any], position: int, *, before: bool, limit: int) -> list[str]:
    indexes = range(position - 1, -1, -1) if before else range(position + 1, len(sequence))
    found: list[str] = []
    for index in indexes:
        block = sequence[index]
        if isinstance(block, dict) and block.get("type") in {"image_ref", "table_ref"}:
            break
        text = _block_text(block)
        if text:
            found.append(text)
        if len(found) >= limit:
            break
    if before:
        found.reverse()
    return found


def collect_visual_refs(report_data: dict[str, Any], context_blocks: int) -> list[RefInfo]:
    refs: list[RefInfo] = []
    for sequence_name in ("main_content", "appendix_content"):
        sequence = report_data.get(sequence_name) or []
        if not isinstance(sequence, list):
            continue
        latest_heading = ""
        for position, block in enumerate(sequence):
            if not isinstance(block, dict):
                continue
            if block.get("type") == "heading":
                latest_heading = _clean_text(block.get("text"))
                continue
            kind = block.get("type")
            if kind not in {"image_ref", "table_ref"}:
                continue
            visual_id = _clean_text(block.get("id"))
            if not visual_id:
                continue
            refs.append(
                RefInfo(
                    visual_id=visual_id,
                    visual_type="figure" if kind == "image_ref" else "table",
                    sequence_name=sequence_name,
                    position=position,
                    original_context=_clean_text(block.get("context")),
                    section_title=latest_heading,
                    context_before=_nearby_text(sequence, position, before=True, limit=context_blocks),
                    context_after=_nearby_text(sequence, position, before=False, limit=context_blocks),
                )
            )
    return refs


def normalize_bbox(bbox: Any) -> list[float]:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return []
    try:
        values = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return []
    if max(abs(value) for value in values) > 2.0:
        values = [value / 1000.0 for value in values]
    x0, y0, x1, y1 = values
    if x1 <= x0 or y1 <= y0:
        return []
    return [max(0.0, x0), max(0.0, y0), min(1.0, x1), min(1.0, y1)]


def _find_pdf(report_dir: Path) -> Path | None:
    preferred = report_dir / f"{report_dir.name}_origin.pdf"
    if preferred.is_file():
        return preferred
    candidates = sorted(report_dir.glob("*_origin.pdf")) or sorted(report_dir.glob("*.pdf"))
    return candidates[0] if candidates else None


def _find_figure_file(report_dir: Path, visual_id: str, content_item: dict[str, Any]) -> Path | None:
    images_dir = report_dir / "images"
    for suffix in IMAGE_SUFFIXES:
        exact = images_dir / f"{visual_id}{suffix}"
        if exact.is_file():
            return exact
    candidates: list[Path] = []
    for suffix in IMAGE_SUFFIXES:
        candidates.extend(images_dir.glob(f"{visual_id}_*{suffix}"))
    if candidates:
        return sorted(candidates)[0]
    relative = content_item.get("img_path")
    if isinstance(relative, str):
        fallback = report_dir / relative
        if fallback.is_file():
            return fallback
    return None


def _find_existing_table_file(report_dir: Path, content_item: dict[str, Any]) -> Path | None:
    relative = content_item.get("img_path")
    if isinstance(relative, str):
        path = report_dir / relative
        if path.is_file():
            return path
    return None


def _content_text(item: dict[str, Any]) -> str:
    for key in ("text", "content"):
        text = _clean_text(item.get(key))
        if text:
            return text
    return ""


def _rect_gap(first: list[float], second: list[float]) -> tuple[float, float]:
    """Return normalized horizontal and vertical gaps between two rectangles."""
    ax0, ay0, ax1, ay1 = first
    bx0, by0, bx1, by1 = second
    gap_x = max(0.0, ax0 - bx1, bx0 - ax1)
    gap_y = max(0.0, ay0 - by1, by0 - ay1)
    return gap_x, gap_y


def _nearby_page_text(
    items: list[dict[str, Any]],
    target_bbox: list[float],
    *,
    exclude: Iterable[str] = (),
    other_visual_bboxes: Iterable[list[float]] = (),
    max_items: int = 8,
    horizontal_margin: float = 0.02,
    vertical_margin: float = 0.10,
) -> list[str]:
    """Select bbox-local labels/notes instead of sending all text on the page."""
    if not target_bbox or max_items <= 0:
        return []
    excluded = {_clean_text(text).casefold() for text in exclude if _clean_text(text)}
    competing_bboxes = [bbox for bbox in other_visual_bboxes if bbox]
    lower_source_edges: list[float] = []
    for item in items:
        text = _content_text(item)
        item_bbox = normalize_bbox(item.get("bbox"))
        if not text or not item_bbox or not SOURCE_RE.search(text) or item_bbox[1] < target_bbox[3]:
            continue
        gap_x, _ = _rect_gap(target_bbox, item_bbox)
        overlap = max(0.0, min(target_bbox[2], item_bbox[2]) - max(target_bbox[0], item_bbox[0]))
        base = min(target_bbox[2] - target_bbox[0], item_bbox[2] - item_bbox[0])
        ratio = overlap / base if base > 0 else 0.0
        if ratio >= 0.25 or gap_x <= horizontal_margin:
            lower_source_edges.append(item_bbox[3])
    lower_source_edge = min(lower_source_edges) if lower_source_edges else None
    candidates: list[tuple[float, int, str]] = []
    for order, item in enumerate(items):
        if item.get("type") not in {"text", "header", "footer", "list"}:
            continue
        text = _content_text(item)
        item_bbox = normalize_bbox(item.get("bbox"))
        if not text or not item_bbox or text.casefold() in excluded:
            continue
        if lower_source_edge is not None and item_bbox[1] > lower_source_edge + 0.005:
            continue
        if text.casefold() == "equity research" or LOW_VALUE_PAGE_TEXT_RE.search(text):
            continue
        gap_x, gap_y = _rect_gap(target_bbox, item_bbox)
        horizontal_overlap = max(0.0, min(target_bbox[2], item_bbox[2]) - max(target_bbox[0], item_bbox[0]))
        overlap_base = min(target_bbox[2] - target_bbox[0], item_bbox[2] - item_bbox[0])
        overlap_ratio = horizontal_overlap / overlap_base if overlap_base > 0 else 0.0
        is_source = bool(SOURCE_RE.search(text))
        # Adjacent newspaper-style columns can be only a few percent apart.
        # Ordinary context must overlap the target horizontally; only a source
        # note may use the small horizontal tolerance.
        if (overlap_ratio < 0.25 and not (is_source and gap_x <= horizontal_margin)) or gap_y > vertical_margin:
            continue
        target_affinity = gap_y * 3.0 + gap_x - overlap_ratio * 0.10
        belongs_to_other_visual = False
        for other_bbox in competing_bboxes:
            other_gap_x, other_gap_y = _rect_gap(other_bbox, item_bbox)
            other_overlap = max(0.0, min(other_bbox[2], item_bbox[2]) - max(other_bbox[0], item_bbox[0]))
            other_base = min(other_bbox[2] - other_bbox[0], item_bbox[2] - item_bbox[0])
            other_ratio = other_overlap / other_base if other_base > 0 else 0.0
            other_affinity = other_gap_y * 3.0 + other_gap_x - other_ratio * 0.10
            if other_ratio >= 0.25 and other_affinity + 0.01 < target_affinity:
                belongs_to_other_visual = True
                break
        if belongs_to_other_visual:
            continue
        # Prefer text directly above/below the target. Source/as-of notes get a
        # small priority boost because they are often just outside the crop.
        priority = gap_y * 3.0 + gap_x
        if is_source:
            priority -= 0.05
        candidates.append((priority, order, text))
    candidates.sort(key=lambda value: (value[0], value[1]))
    return _deduplicate(text for _, _, text in candidates)[:max_items]


def _neighboring_visuals(content: list[Any], target: dict[str, Any], page_idx: int) -> list[dict[str, Any]]:
    """Return the closest visual before/after the target on the same page.

    MinerU sometimes splits one logical financial panel into adjacent table
    blocks. The neighboring block may carry year headers missing from the
    target crop, so expose a bounded structured view to the model.
    """
    try:
        target_index = next(index for index, item in enumerate(content) if item is target)
    except StopIteration:
        return []
    result: list[dict[str, Any]] = []
    target_type = target.get("type")
    for direction, label in ((-1, "before"), (1, "after")):
        index = target_index + direction
        while 0 <= index < len(content):
            item = content[index]
            index += direction
            if not isinstance(item, dict):
                continue
            candidate_page = item.get("page_idx")
            if isinstance(candidate_page, int) and candidate_page != page_idx:
                break
            if item.get("type") not in {"image", "table"}:
                continue
            if item.get("type") != target_type:
                continue
            caption_key = "image_caption" if item.get("type") == "image" else "table_caption"
            footnote_key = "image_footnote" if item.get("type") == "image" else "table_footnote"
            body = _clean_text(item.get("table_body"))
            result.append(
                {
                    "relative_position": label,
                    "type": item.get("type"),
                    "bbox_normalized": normalize_bbox(item.get("bbox")),
                    "caption": _clean_text(item.get(caption_key)),
                    "footnote": _clean_text(item.get(footnote_key)),
                    "table_body_excerpt": body[:8_000],
                    "table_body_truncated": len(body) > 8_000,
                }
            )
            break
    return result


class PdfAssetRenderer:
    """Render annotated pages and bbox crops with one open PDF handle."""

    def __init__(self, pdf_path: Path | None, output_dir: Path, page_dpi: int, crop_dpi: int, padding: float):
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.page_dpi = page_dpi
        self.crop_dpi = crop_dpi
        self.padding = padding
        self.document = None
        self.error = ""
        if pdf_path is None:
            self.error = "origin PDF not found"
            return
        try:
            import fitz  # type: ignore

            self.fitz = fitz
            self.document = fitz.open(pdf_path)
        except Exception as exc:  # dependency or malformed PDF
            self.error = f"PDF rendering unavailable: {type(exc).__name__}: {exc}"

    def close(self) -> None:
        if self.document is not None:
            self.document.close()

    def _page_and_rect(self, page_idx: int, bbox: list[float]):
        if self.document is None or not bbox or page_idx < 0 or page_idx >= len(self.document):
            return None, None
        page = self.document[page_idx]
        x0, y0, x1, y1 = bbox
        pad = self.padding
        x0, y0 = max(0.0, x0 - pad), max(0.0, y0 - pad)
        x1, y1 = min(1.0, x1 + pad), min(1.0, y1 + pad)
        rect = self.fitz.Rect(
            page.rect.x0 + x0 * page.rect.width,
            page.rect.y0 + y0 * page.rect.height,
            page.rect.x0 + x1 * page.rect.width,
            page.rect.y0 + y1 * page.rect.height,
        )
        return page, rect

    @staticmethod
    def _pixmap_to_image(pixmap) -> Image.Image:
        mode = "RGBA" if pixmap.alpha else "RGB"
        image = Image.frombytes(mode, (pixmap.width, pixmap.height), pixmap.samples)
        return image.convert("RGB")

    def render_crop(self, page_idx: int, bbox: list[float], output_path: Path, *, dpi: int | None = None) -> Path | None:
        page, rect = self._page_and_rect(page_idx, bbox)
        if page is None:
            return None
        resolution = self.crop_dpi if dpi is None else dpi
        matrix = self.fitz.Matrix(resolution / 72.0, resolution / 72.0)
        pixmap = page.get_pixmap(matrix=matrix, clip=rect, alpha=False)
        image = self._pixmap_to_image(pixmap)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, format="JPEG", quality=95, subsampling=0, optimize=True)
        return output_path

    def render_annotated_page(self, page_idx: int, bbox: list[float], output_path: Path) -> Path | None:
        if self.document is None or not bbox or page_idx < 0 or page_idx >= len(self.document):
            return None
        page = self.document[page_idx]
        matrix = self.fitz.Matrix(self.page_dpi / 72.0, self.page_dpi / 72.0)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        image = self._pixmap_to_image(pixmap)
        x0, y0, x1, y1 = bbox
        draw = ImageDraw.Draw(image)
        rect = [x0 * image.width, y0 * image.height, x1 * image.width, y1 * image.height]
        line_width = max(4, round(min(image.size) / 250))
        for offset in range(line_width):
            draw.rectangle([rect[0] - offset, rect[1] - offset, rect[2] + offset, rect[3] + offset], outline=(220, 30, 30))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, format="JPEG", quality=88, subsampling=1, optimize=True)
        return output_path


def _deduplicate(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = _clean_text(value)
        key = clean.casefold()
        if clean and key not in seen:
            seen.add(key)
            result.append(clean)
    return result


def build_evidence(
    report_dir: Path,
    evidence_dir: Path,
    *,
    context_blocks: int = 4,
    nearby_text_items: int = 8,
    nearby_horizontal_margin: float = 0.02,
    nearby_vertical_margin: float = 0.10,
    page_dpi: int = 120,
    crop_dpi: int = 220,
    bbox_padding: float = 0.004,
) -> tuple[list[VisualEvidence], list[str]]:
    report_id = report_dir.name
    report_json_path = report_dir / f"{report_id}.json"
    content_path = report_dir / f"{report_id}_content_list.json"
    errors: list[str] = []
    report = json.loads(report_json_path.read_text(encoding="utf-8"))
    content = json.loads(content_path.read_text(encoding="utf-8"))
    report_data = report.get("data") or {}
    refs = collect_visual_refs(report_data, context_blocks)
    figures = [item for item in content if isinstance(item, dict) and item.get("type") == "image"]
    tables = [item for item in content if isinstance(item, dict) and item.get("type") == "table"]
    ref_figures = [item for item in refs if item.visual_type == "figure"]
    ref_tables = [item for item in refs if item.visual_type == "table"]
    if len(ref_figures) != len(figures):
        errors.append(f"figure mapping count mismatch: refs={len(ref_figures)}, content_list={len(figures)}")
    if len(ref_tables) != len(tables):
        errors.append(f"table mapping count mismatch: refs={len(ref_tables)}, content_list={len(tables)}")

    content_by_ref: dict[str, dict[str, Any]] = {}
    for ref, item in zip(ref_figures, figures):
        content_by_ref[ref.visual_id] = item
    for ref, item in zip(ref_tables, tables):
        content_by_ref[ref.visual_id] = item

    page_items: dict[int, list[dict[str, Any]]] = {}
    for item in content:
        if isinstance(item, dict) and isinstance(item.get("page_idx"), int):
            page_items.setdefault(item["page_idx"], []).append(item)

    renderer = PdfAssetRenderer(_find_pdf(report_dir), evidence_dir, page_dpi, crop_dpi, bbox_padding)
    if renderer.error:
        errors.append(renderer.error)
    evidences: list[VisualEvidence] = []
    try:
        for ref in refs:
            item = content_by_ref.get(ref.visual_id)
            if item is None:
                errors.append(f"no content_list mapping for {ref.visual_id}")
                continue
            page_idx = item.get("page_idx")
            if not isinstance(page_idx, int):
                errors.append(f"invalid page_idx for {ref.visual_id}")
                continue
            bbox = normalize_bbox(item.get("bbox"))
            caption_key = "image_caption" if ref.visual_type == "figure" else "table_caption"
            footnote_key = "image_footnote" if ref.visual_type == "figure" else "table_footnote"
            original_caption = _clean_text(item.get(caption_key)) or ref.original_context
            original_footnote = _clean_text(item.get(footnote_key))
            context_before = _deduplicate(ref.context_before)
            context_after = _deduplicate(
                text for text in ref.context_after
                if _clean_text(text).casefold() not in {_clean_text(x).casefold() for x in context_before}
            )
            nearby_page_text = _nearby_page_text(
                page_items.get(page_idx, []),
                bbox,
                exclude=[
                    original_caption,
                    original_footnote,
                    ref.section_title,
                    *context_before,
                    *context_after,
                ],
                other_visual_bboxes=(
                    normalize_bbox(page_item.get("bbox"))
                    for page_item in page_items.get(page_idx, [])
                    if page_item is not item and page_item.get("type") in {"image", "table"}
                ),
                max_items=nearby_text_items,
                horizontal_margin=nearby_horizontal_margin,
                vertical_margin=nearby_vertical_margin,
            )
            asset_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", ref.visual_id)
            page_image = renderer.render_annotated_page(
                page_idx, bbox, evidence_dir / "pages" / f"{asset_stem}_page_{page_idx:04d}.jpg"
            )
            target_image: Path | None
            warnings: list[str] = []
            if ref.visual_type == "figure":
                target_image = _find_figure_file(report_dir, ref.visual_id, item)
                if target_image is None:
                    target_image = renderer.render_crop(
                        page_idx, bbox, evidence_dir / "targets" / f"{asset_stem}_figure.jpg"
                    )
                    warnings.append("named figure file missing; target recropped from PDF bbox")
            else:
                target_image = renderer.render_crop(
                    page_idx, bbox, evidence_dir / "targets" / f"{asset_stem}_table.jpg"
                )
                if target_image is None:
                    target_image = _find_existing_table_file(report_dir, item)
                    if target_image is not None:
                        warnings.append("PDF renderer unavailable; using existing extracted table image")
            if target_image is None:
                warnings.append("target image unavailable")
            if not bbox:
                warnings.append("invalid bbox")
            evidences.append(
                VisualEvidence(
                    report_id=report_id,
                    visual_id=ref.visual_id,
                    visual_type=ref.visual_type,
                    report_title=_clean_text(report.get("raw_title") or report_data.get("title")),
                    author_org=_clean_text(report.get("institution")),
                    publish_time=_clean_text(report.get("raw_date") or report.get("release_timestamp"))[:10],
                    security_code=_clean_text(report.get("security_code")),
                    page_idx=page_idx,
                    bbox=bbox,
                    original_caption=original_caption,
                    original_footnote=original_footnote,
                    section_title=ref.section_title,
                    context_before=context_before,
                    context_after=context_after,
                    nearby_page_text=nearby_page_text,
                    neighboring_visuals=_neighboring_visuals(content, item, page_idx),
                    table_body=_clean_text(item.get("table_body")),
                    target_image=target_image,
                    page_image=page_image,
                    mapping_warnings=warnings,
                )
            )
    finally:
        renderer.close()
    return evidences, errors


def image_quality(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {"readable": False, "width": 0, "height": 0, "format": "", "hard_filter_reason": "image file missing"}
    try:
        with Image.open(path) as raw:
            image = ImageOps.exif_transpose(raw)
            image.load()
            width, height = image.size
            hard_reason = ""
            if width < 64 or height < 48 or width * height < 4096:
                hard_reason = f"image too small: {width}x{height}"
            return {
                "readable": True,
                "width": width,
                "height": height,
                "format": raw.format or "",
                "bytes": path.stat().st_size,
                "hard_filter_reason": hard_reason,
            }
    except Exception as exc:
        return {
            "readable": False,
            "width": 0,
            "height": 0,
            "format": "",
            "hard_filter_reason": f"cannot decode image: {type(exc).__name__}: {exc}",
        }
