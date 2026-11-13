from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cir_qdiag_openai_cirr.schemas import CIRRSampleRecord, ImageRef, SampleMetadata


class CIRRDataError(ValueError):
    pass


def load_cirr_train(
    annotation_path: str | Path,
    image_split_path: str | Path,
    image_root: str | Path,
    *,
    verify_images: bool = True,
) -> list[CIRRSampleRecord]:
    """Load official CIRR train triplets in annotation order."""

    annotation_path = Path(annotation_path)
    image_split_path = Path(image_split_path)
    image_root = Path(image_root)
    annotations = _load_json(annotation_path)
    image_map = _load_json(image_split_path)
    if not isinstance(annotations, list):
        raise CIRRDataError(f"expected annotation list: {annotation_path}")
    if not isinstance(image_map, dict):
        raise CIRRDataError(f"expected image mapping object: {image_split_path}")

    records: list[CIRRSampleRecord] = []
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

        reference_id = _required_string(item, "reference", pair_id)
        target_id = _required_string(item, "target_hard", pair_id)
        positive_text = _required_string(item, "caption", pair_id).strip()
        if reference_id == target_id:
            raise CIRRDataError(f"pair {pair_id} uses the same reference and target")
        if not positive_text:
            raise CIRRDataError(f"pair {pair_id} has an empty caption")

        image_set_payload = item.get("img_set")
        if not isinstance(image_set_payload, dict):
            raise CIRRDataError(f"pair {pair_id} has an invalid img_set")
        members = image_set_payload.get("members")
        if not isinstance(members, list) or not all(
            isinstance(member, str) and member for member in members
        ):
            raise CIRRDataError(f"pair {pair_id} has invalid img_set members")

        records.append(
            CIRRSampleRecord(
                sample_id=f"cirr_train_{pair_id}",
                pair_id=pair_id,
                reference=ImageRef(
                    image_id=reference_id,
                    image_path=str(
                        _resolve_image(
                            reference_id,
                            image_map,
                            image_root,
                            verify_images=verify_images,
                        )
                    ),
                ),
                target=ImageRef(
                    image_id=target_id,
                    image_path=str(
                        _resolve_image(
                            target_id,
                            image_map,
                            image_root,
                            verify_images=verify_images,
                        )
                    ),
                ),
                positive_text=" ".join(positive_text.split()),
                image_set=list(members),
                metadata=SampleMetadata(stage="source", operational_status="ok"),
            )
        )
    return records


def _required_string(item: dict[str, Any], key: str, pair_id: int) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise CIRRDataError(f"pair {pair_id} has invalid {key}: {value!r}")
    return value.strip()


def _resolve_image(
    image_id: str,
    image_map: dict[str, Any],
    image_root: Path,
    *,
    verify_images: bool,
) -> Path:
    mapped = image_map.get(image_id)
    if not isinstance(mapped, str) or not mapped:
        raise CIRRDataError(f"image id absent from split mapping: {image_id}")
    relative = Path(mapped.removeprefix("./"))
    if relative.is_absolute() or ".." in relative.parts:
        raise CIRRDataError(f"unsafe image path for {image_id}: {mapped}")
    root = image_root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise CIRRDataError(f"unsafe image path for {image_id}: {mapped}")
    if verify_images and not path.is_file():
        raise CIRRDataError(f"missing image for {image_id}: {path}")
    return path


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise CIRRDataError(f"JSON file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CIRRDataError(f"invalid JSON in {path}: {exc}") from exc
