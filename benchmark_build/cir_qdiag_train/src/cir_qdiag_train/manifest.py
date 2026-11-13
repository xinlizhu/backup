from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HardNegativeText(StrictModel):
    negative_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    semantic_target: Literal["object", "attribute", "predicate"]
    edit_operation: Literal["add", "delete", "substitute"]
    construction_mode: Literal["text_only_editable", "vision_grounded"]


class TrainManifestRecord(StrictModel):
    sample_id: str = Field(min_length=1)
    pair_id: int = Field(ge=0)
    reference_id: str = Field(min_length=1)
    reference_path: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    target_path: str = Field(min_length=1)
    positive_text: str = Field(min_length=1)
    image_set: list[str] = Field(default_factory=list)
    negatives: list[HardNegativeText] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_negatives(self) -> "TrainManifestRecord":
        ids = [item.negative_id for item in self.negatives]
        if len(ids) != len(set(ids)):
            raise ValueError("negative ids must be unique within a sample")
        texts = [_normalize(item.text) for item in self.negatives]
        if len(texts) != len(set(texts)):
            raise ValueError("negative texts must be unique within a sample")
        return self


def build_train_manifest(
    *,
    annotation_path: str | Path,
    image_split_path: str | Path,
    image_root: str | Path,
    negative_jsonl_paths: Iterable[str | Path],
    output_path: str | Path,
    require_all_samples_have_negatives: bool = False,
    only_samples_with_negatives: bool = False,
) -> dict[str, object]:
    annotation_path = Path(annotation_path)
    image_split_path = Path(image_split_path)
    image_root = Path(image_root).resolve()
    annotations = _load_json(annotation_path)
    image_map = _load_json(image_split_path)
    if not isinstance(annotations, list):
        raise ValueError(f"expected a list in {annotation_path}")
    if not isinstance(image_map, dict):
        raise ValueError(f"expected an object in {image_split_path}")

    annotation_by_pair: dict[int, dict] = {}
    for index, row in enumerate(annotations):
        if not isinstance(row, dict):
            raise ValueError(f"annotation {index} is not an object")
        pair_id = row.get("pairid")
        if not isinstance(pair_id, int) or isinstance(pair_id, bool):
            raise ValueError(f"annotation {index} has invalid pairid")
        if pair_id in annotation_by_pair:
            raise ValueError(f"duplicate train pairid: {pair_id}")
        annotation_by_pair[pair_id] = row

    negatives_by_pair = _load_generated_negatives(
        negative_jsonl_paths,
        annotation_by_pair,
    )
    if require_all_samples_have_negatives:
        missing = sorted(set(annotation_by_pair) - set(negatives_by_pair))
        if missing:
            raise ValueError(
                f"{len(missing)} CIRR train samples have no generated negatives; "
                f"first missing pair ids: {missing[:10]}"
            )

    records: list[TrainManifestRecord] = []
    for pair_id, annotation in annotation_by_pair.items():
        if only_samples_with_negatives and pair_id not in negatives_by_pair:
            continue
        reference_id = _required_string(annotation, "reference", pair_id)
        target_id = annotation.get("target_hard", annotation.get("target"))
        if not isinstance(target_id, str) or not target_id:
            raise ValueError(f"pair {pair_id} has no target")
        positive_text = _required_string(annotation, "caption", pair_id).strip()
        image_set_payload = annotation.get("img_set", {})
        image_set = (
            image_set_payload.get("members", [])
            if isinstance(image_set_payload, dict)
            else []
        )
        if not isinstance(image_set, list) or not all(
            isinstance(item, str) for item in image_set
        ):
            raise ValueError(f"pair {pair_id} has an invalid image set")
        records.append(
            TrainManifestRecord(
                sample_id=f"cirr_train_{pair_id}",
                pair_id=pair_id,
                reference_id=reference_id,
                reference_path=str(_resolve_image(reference_id, image_map, image_root)),
                target_id=target_id,
                target_path=str(_resolve_image(target_id, image_map, image_root)),
                positive_text=positive_text,
                image_set=image_set,
                negatives=negatives_by_pair.get(pair_id, []),
            )
        )

    output_path = Path(output_path)
    _write_jsonl_atomic(output_path, records)
    operations = Counter(
        negative.edit_operation for record in records for negative in record.negatives
    )
    targets = Counter(
        negative.semantic_target for record in records for negative in record.negatives
    )
    modes = Counter(
        negative.construction_mode for record in records for negative in record.negatives
    )
    summary: dict[str, object] = {
        "num_train_samples": len(records),
        "only_samples_with_negatives": only_samples_with_negatives,
        "samples_with_negatives": sum(bool(record.negatives) for record in records),
        "samples_without_negatives": sum(not record.negatives for record in records),
        "num_negatives": sum(len(record.negatives) for record in records),
        "negatives_by_operation": dict(sorted(operations.items())),
        "negatives_by_semantic_target": dict(sorted(targets.items())),
        "negatives_by_construction_mode": dict(sorted(modes.items())),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def read_manifest(path: str | Path) -> list[TrainManifestRecord]:
    records: list[TrainManifestRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(TrainManifestRecord.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"invalid manifest row {path}:{line_number}: {exc}") from exc
    return records


def _load_generated_negatives(
    paths: Iterable[str | Path],
    annotation_by_pair: dict[int, dict],
) -> dict[int, list[HardNegativeText]]:
    by_pair: dict[int, list[HardNegativeText]] = {}
    seen_ids: dict[str, HardNegativeText] = {}
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
                split = row.get("split")
                sample_id = str(row.get("sample_id", ""))
                if split != "train" or not sample_id.startswith("cirr_train_"):
                    raise ValueError(
                        f"{path}:{line_number} is not a CIRR train negative record "
                        f"(split={split!r}, sample_id={sample_id!r}); never train with val output"
                    )
                pair_id = row.get("pair_id")
                if not isinstance(pair_id, int) or pair_id not in annotation_by_pair:
                    raise ValueError(f"{path}:{line_number} has unknown train pair_id {pair_id!r}")
                annotation = annotation_by_pair[pair_id]
                expected_positive = _required_string(annotation, "caption", pair_id)
                if _normalize(str(row.get("positive_text", ""))) != _normalize(
                    expected_positive
                ):
                    raise ValueError(
                        f"{path}:{line_number} positive text does not match CIRR train pair "
                        f"{pair_id}"
                    )
                if row.get("reference", {}).get("image_id") != annotation.get("reference"):
                    raise ValueError(f"{path}:{line_number} reference mismatch for pair {pair_id}")
                expected_target = annotation.get("target_hard", annotation.get("target"))
                if row.get("target", {}).get("image_id") != expected_target:
                    raise ValueError(f"{path}:{line_number} target mismatch for pair {pair_id}")

                candidates = row.get("negatives")
                if not isinstance(candidates, list):
                    candidates = [
                        candidate
                        for candidate in row.get("candidates", [])
                        if isinstance(candidate, dict)
                        and candidate.get("status") == "accepted"
                    ]
                for candidate in candidates:
                    if not isinstance(candidate, dict):
                        raise ValueError(
                            f"{path}:{line_number} contains a non-object negative"
                        )
                    if candidate.get("status") != "accepted":
                        continue
                    negative = HardNegativeText(
                        negative_id=candidate["negative_id"],
                        text=candidate["text"],
                        semantic_target=candidate["semantic_target"],
                        edit_operation=candidate["edit_operation"],
                        construction_mode=candidate["construction_mode"],
                    )
                    previous = seen_ids.get(negative.negative_id)
                    if previous is not None:
                        if previous != negative:
                            raise ValueError(
                                f"conflicting duplicate negative id: {negative.negative_id}"
                            )
                        continue
                    seen_ids[negative.negative_id] = negative
                    by_pair.setdefault(pair_id, []).append(negative)
    return by_pair


def _resolve_image(image_id: str, image_map: dict, image_root: Path) -> Path:
    mapped = image_map.get(image_id)
    if not isinstance(mapped, str) or not mapped:
        raise ValueError(f"image id is absent from the train split map: {image_id}")
    relative = Path(mapped.removeprefix("./"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe image path for {image_id}: {mapped}")
    path = (image_root / relative).resolve()
    if not path.is_relative_to(image_root):
        raise ValueError(f"unsafe image path for {image_id}: {mapped}")
    if not path.is_file():
        raise FileNotFoundError(f"missing CIRR image: {path}")
    return path


def _required_string(row: dict, key: str, pair_id: int) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"pair {pair_id} has invalid {key}")
    return value


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"data file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc


def _write_jsonl_atomic(path: Path, records: Iterable[TrainManifestRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())
