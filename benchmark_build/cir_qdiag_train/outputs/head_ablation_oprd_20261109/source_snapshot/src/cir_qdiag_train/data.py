from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from cir_qdiag_train.manifest import HardNegativeText, TrainManifestRecord, read_manifest


class CIRRHardNegativeDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        negative_mode: str,
        max_non_delete_negatives_per_sample: int,
        max_delete_negatives_per_sample: int,
        seed: int,
        require_usable_negatives: bool = False,
        candidate_image_mode: str = "none",
        train_image_split_path: str | Path | None = None,
        image_root: str | Path | None = None,
        max_candidate_images_per_sample: int = 4,
    ) -> None:
        self.records = read_manifest(manifest_path)
        if negative_mode not in {"none", "add_substitute", "add_substitute_delete"}:
            raise ValueError(f"unsupported negative_mode: {negative_mode}")
        self.negative_mode = negative_mode
        self.max_non_delete_negatives_per_sample = max_non_delete_negatives_per_sample
        self.max_delete_negatives_per_sample = max_delete_negatives_per_sample
        self.seed = seed
        self.epoch = 0
        if candidate_image_mode not in {"none", "image_set"}:
            raise ValueError(f"unsupported candidate_image_mode: {candidate_image_mode}")
        self.candidate_image_mode = candidate_image_mode
        self.max_candidate_images_per_sample = max_candidate_images_per_sample
        self.image_map: dict[str, str] = {}
        self.image_root: Path | None = None
        if candidate_image_mode == "image_set":
            if train_image_split_path is None or image_root is None:
                raise ValueError(
                    "image_set candidates require train_image_split_path and image_root"
                )
            payload = json.loads(Path(train_image_split_path).read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in payload.items()
            ):
                raise TypeError("invalid train image split map")
            self.image_map = payload
            self.image_root = Path(image_root).resolve()
        if require_usable_negatives:
            missing = [
                record.sample_id
                for record in self.records
                if not self._usable_negatives(record)
            ]
            if missing:
                raise ValueError(
                    "hard-only training requires every manifest record to have at least "
                    f"one usable negative; missing={len(missing)} first={missing[:10]}"
                )

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        candidate_ids = self._candidate_ids(record)
        candidate_images = [
            _load_image(self._resolve_candidate_image(candidate_id))
            for candidate_id in candidate_ids
        ]
        if self.negative_mode == "none":
            negatives = []
            return {
                "sample_id": record.sample_id,
                "target_id": record.target_id,
                "reference_image": _load_image(record.reference_path),
                "target_image": _load_image(record.target_path),
                "positive_text": record.positive_text,
                "negatives": negatives,
                "candidate_ids": candidate_ids,
                "candidate_images": candidate_images,
            }
        negatives = self._usable_negatives(record)
        non_delete = [
            negative
            for negative in negatives
            if negative.edit_operation != "delete"
        ]
        deletes = [
            negative for negative in negatives if negative.edit_operation == "delete"
        ]
        non_delete = sorted(
            non_delete,
            key=lambda negative: _stable_order_key(
                self.seed,
                self.epoch,
                record.sample_id,
                negative.negative_id,
            ),
        )[: self.max_non_delete_negatives_per_sample]
        deletes = sorted(
            deletes,
            key=lambda negative: _stable_order_key(
                self.seed,
                self.epoch,
                record.sample_id,
                negative.negative_id,
            ),
        )[: self.max_delete_negatives_per_sample]
        if self.negative_mode == "add_substitute":
            negatives = non_delete
        else:
            negatives = non_delete + deletes
        return {
            "sample_id": record.sample_id,
            "target_id": record.target_id,
            "reference_image": _load_image(record.reference_path),
            "target_image": _load_image(record.target_path),
            "positive_text": record.positive_text,
            "negatives": negatives,
            "candidate_ids": candidate_ids,
            "candidate_images": candidate_images,
        }

    def _candidate_ids(self, record: TrainManifestRecord) -> list[str]:
        if self.candidate_image_mode == "none":
            return []
        excluded = {record.reference_id, record.target_id}
        candidates = sorted(set(record.image_set) - excluded)
        return sorted(
            candidates,
            key=lambda candidate_id: _stable_order_key(
                self.seed,
                self.epoch,
                record.sample_id,
                candidate_id,
            ),
        )[: self.max_candidate_images_per_sample]

    def _resolve_candidate_image(self, image_id: str) -> Path:
        if self.image_root is None:
            raise RuntimeError("candidate image root is not configured")
        mapped = self.image_map.get(image_id)
        if not isinstance(mapped, str) or not mapped:
            raise ValueError(f"candidate image is absent from train split map: {image_id}")
        relative = Path(mapped.removeprefix("./"))
        path = (self.image_root / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(
            self.image_root
        ):
            raise ValueError(f"unsafe candidate image path: {mapped}")
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def _usable_negatives(
        self,
        record: TrainManifestRecord,
    ) -> list[HardNegativeText]:
        if self.negative_mode == "none":
            return []
        if self.negative_mode == "add_substitute":
            return [
                negative
                for negative in record.negatives
                if negative.edit_operation != "delete"
            ]
        return list(record.negatives)


class CIRRTrainCollator:
    def __init__(self, processor) -> None:
        self.processor = processor

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        reference = self.processor(
            images=[example["reference_image"] for example in examples],
            return_tensors="pt",
        )
        target = self.processor(
            images=[example["target_image"] for example in examples],
            return_tensors="pt",
        )
        positive = self.processor(
            text=[example["positive_text"] for example in examples],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        negative_texts: list[str] = []
        negative_owner: list[int] = []
        negative_semantic_class: list[int] = []
        negative_metadata: list[HardNegativeText] = []
        semantic_class = {"object": 0, "attribute": 1, "predicate": 2}
        for owner, example in enumerate(examples):
            for negative in example["negatives"]:
                negative_texts.append(negative.text)
                negative_owner.append(owner)
                negative_semantic_class.append(
                    semantic_class[negative.semantic_target]
                )
                negative_metadata.append(negative)
        negative = (
            self.processor(
                text=negative_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            if negative_texts
            else None
        )
        candidate_images: list[Image.Image] = []
        candidate_owner: list[int] = []
        candidate_ids: list[str] = []
        for owner, example in enumerate(examples):
            candidate_images.extend(example["candidate_images"])
            candidate_owner.extend([owner] * len(example["candidate_images"]))
            candidate_ids.extend(example["candidate_ids"])
        candidate = (
            self.processor(images=candidate_images, return_tensors="pt")
            if candidate_images
            else None
        )
        return {
            "sample_ids": [example["sample_id"] for example in examples],
            "target_ids": [example["target_id"] for example in examples],
            "reference_pixel_values": reference["pixel_values"],
            "target_pixel_values": target["pixel_values"],
            "positive_input_ids": positive["input_ids"],
            "positive_attention_mask": positive.get("attention_mask"),
            "negative_input_ids": None if negative is None else negative["input_ids"],
            "negative_attention_mask": (
                None if negative is None else negative.get("attention_mask")
            ),
            "negative_owner": torch.tensor(negative_owner, dtype=torch.long),
            "negative_semantic_class": torch.tensor(
                negative_semantic_class, dtype=torch.long
            ),
            "negative_metadata": negative_metadata,
            "candidate_pixel_values": (
                None if candidate is None else candidate["pixel_values"]
            ),
            "candidate_owner": torch.tensor(candidate_owner, dtype=torch.long),
            "candidate_ids": candidate_ids,
        }


def count_usable_negatives(
    records: list[TrainManifestRecord],
    *,
    negative_mode: str,
) -> dict[str, int]:
    counts = {"add": 0, "delete": 0, "substitute": 0}
    if negative_mode == "none":
        return counts
    for record in records:
        for negative in record.negatives:
            if (
                negative_mode == "add_substitute_delete"
                or negative.edit_operation != "delete"
            ):
                counts[negative.edit_operation] += 1
    return counts


def _load_image(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _stable_order_key(*parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
