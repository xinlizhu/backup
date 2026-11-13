from __future__ import annotations

import json
from pathlib import Path

from cir_qdiag.schemas import ImageRef, SampleMetadata, SampleRecord


class CIRRDataError(ValueError):
    pass


def load_cirr_annotations(
    annotation_path: Path,
    image_root: Path,
    split: str = "val",
    image_split_path: Path | None = None,
) -> list[SampleRecord]:
    if split not in {"train", "val"}:
        raise CIRRDataError(
            f"CIR-QDiag construction supports only the train or validation split, not {split}"
        )
    annotation_path = Path(annotation_path)
    image_root = Path(image_root)
    if image_split_path is None:
        image_split_path = _infer_image_split_path(annotation_path, split)
    image_split_path = Path(image_split_path)
    annotations = _load_json(annotation_path)
    image_map = _load_json(image_split_path)
    if not isinstance(annotations, list):
        raise CIRRDataError(f"expected annotation list: {annotation_path}")
    if not isinstance(image_map, dict):
        raise CIRRDataError(f"expected image mapping object: {image_split_path}")

    records: list[SampleRecord] = []
    seen_pair_ids: set[int] = set()
    for index, item in enumerate(annotations):
        if not isinstance(item, dict):
            raise CIRRDataError(f"record {index} must be a JSON object")
        pair_id = item.get("pairid")
        if not isinstance(pair_id, int) or isinstance(pair_id, bool):
            raise CIRRDataError(f"record {index} has invalid pairid: {pair_id!r}")
        if pair_id in seen_pair_ids:
            raise CIRRDataError(f"duplicate pairid: {pair_id}")
        seen_pair_ids.add(pair_id)
        reference_id = item.get("reference")
        target_id = item.get("target_hard", item.get("target"))
        caption = item.get("caption")
        if not isinstance(reference_id, str) or not isinstance(target_id, str):
            raise CIRRDataError(f"pair {pair_id} lacks reference or validation target")
        positive_text = caption.strip() if isinstance(caption, str) else ""
        if reference_id == target_id:
            raise CIRRDataError(f"pair {pair_id} uses the same reference and target")
        if not positive_text:
            raise CIRRDataError(f"pair {pair_id} has an empty caption")
        reference_path = _resolve_image(reference_id, image_map, image_root)
        target_path = _resolve_image(target_id, image_map, image_root)
        image_set_payload = item.get("img_set", {})
        image_set = (
            image_set_payload.get("members", []) if isinstance(image_set_payload, dict) else []
        )
        if not isinstance(image_set, list) or not all(
            isinstance(member, str) for member in image_set
        ):
            raise CIRRDataError(f"pair {pair_id} has invalid img_set members")
        records.append(
            SampleRecord(
                sample_id=f"cirr_{split}_{pair_id}",
                split=split,
                pair_id=pair_id,
                reference=ImageRef(image_id=reference_id, image_path=str(reference_path)),
                target=ImageRef(image_id=target_id, image_path=str(target_path)),
                positive_text=positive_text,
                image_set=list(image_set),
                metadata=SampleMetadata(
                    stage="source",
                    operational_status="ok",
                ),
            )
        )
    return records


def _resolve_image(image_id: str, image_map: dict[str, str], image_root: Path) -> Path:
    if image_id not in image_map:
        raise CIRRDataError(f"image id absent from split mapping: {image_id}")
    mapped = image_map[image_id]
    if not isinstance(mapped, str) or not mapped:
        raise CIRRDataError(f"invalid image path for {image_id}: {mapped!r}")
    relative = mapped.removeprefix("./")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise CIRRDataError(f"unsafe image path for {image_id}: {relative}")
    root = image_root.resolve()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root):
        raise CIRRDataError(f"unsafe image path for {image_id}: {relative}")
    if not path.is_file():
        raise CIRRDataError(f"missing image for {image_id}: {path}")
    return path


def _infer_image_split_path(annotation_path: Path, split: str) -> Path:
    version = annotation_path.name.removeprefix("cap.").removesuffix(f".{split}.json")
    candidate = annotation_path.parent.parent / "image_splits" / f"split.{version}.{split}.json"
    if not candidate.is_file():
        raise CIRRDataError(f"could not infer image split map: {candidate}")
    return candidate


def _load_json(path: Path):
    if not path.is_file():
        raise CIRRDataError(f"JSON file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CIRRDataError(f"invalid JSON in {path}: {exc}") from exc
