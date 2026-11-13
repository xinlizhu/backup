from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from cir_qdiag_train.manifest import HardNegativeText, read_manifest

SEMANTIC_CLASSES = ("object", "attribute", "predicate")
SEMANTIC_CLASS_TO_ID = {label: index for index, label in enumerate(SEMANTIC_CLASSES)}


class CIRROAPDataset(Dataset):
    """CIRR pairs with balanced Add/Substitute supervision for O/A/P patching."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        max_negatives_per_semantic: int,
        seed: int,
    ) -> None:
        self.records = read_manifest(manifest_path)
        self.max_negatives_per_semantic = max_negatives_per_semantic
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        selected: list[HardNegativeText] = []
        for semantic in SEMANTIC_CLASSES:
            choices = [
                negative
                for negative in record.negatives
                if negative.semantic_target == semantic
                and negative.edit_operation in {"add", "substitute"}
            ]
            choices.sort(
                key=lambda negative: _stable_order_key(
                    self.seed,
                    self.epoch,
                    record.sample_id,
                    negative.negative_id,
                )
            )
            selected.extend(choices[: self.max_negatives_per_semantic])
        return {
            "sample_id": record.sample_id,
            "target_id": record.target_id,
            "reference_image": _load_image(record.reference_path),
            "target_image": _load_image(record.target_path),
            "positive_text": record.positive_text,
            "negatives": selected,
        }


class CIRROAPCollator:
    def __init__(self, processor: Any) -> None:
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
        for owner, example in enumerate(examples):
            for negative in example["negatives"]:
                negative_texts.append(negative.text)
                negative_owner.append(owner)
                negative_semantic_class.append(
                    SEMANTIC_CLASS_TO_ID[negative.semantic_target]
                )
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
        }


def _load_image(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _stable_order_key(*parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
