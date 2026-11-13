from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from cir_qdiag_train.manifest import read_manifest
from cir_qdiag_train.on_policy_config import OnPolicyDistillationConfig


@dataclass(frozen=True)
class CIRRPositiveRecord:
    pair_id: int
    reference_id: str
    reference_path: Path
    target_id: str
    target_path: Path
    text: str


@dataclass(frozen=True)
class NegativeQueryLayout:
    ids: list[str]
    texts: list[str]
    owners: np.ndarray
    indices_by_owner: list[list[int]]


def load_negative_query_layout(
    manifest_path: Path,
    records: list[CIRRPositiveRecord],
) -> NegativeQueryLayout:
    """Align CIR-QDiag q- records to the original CIRR training order."""

    record_by_pair = {record.pair_id: (index, record) for index, record in enumerate(records)}
    manifest_by_pair = {}
    for manifest_record in read_manifest(manifest_path):
        if manifest_record.pair_id in manifest_by_pair:
            raise ValueError(
                f"duplicate q- manifest pair_id: {manifest_record.pair_id}"
            )
        try:
            _, positive = record_by_pair[manifest_record.pair_id]
        except KeyError as exc:
            raise ValueError(
                f"q- manifest contains unknown CIRR pair {manifest_record.pair_id}"
            ) from exc
        if (
            manifest_record.reference_id != positive.reference_id
            or manifest_record.target_id != positive.target_id
            or _normalize_text(manifest_record.positive_text)
            != _normalize_text(positive.text)
        ):
            raise ValueError(
                f"q- manifest does not match CIRR pair {manifest_record.pair_id}"
            )
        manifest_by_pair[manifest_record.pair_id] = manifest_record

    ids: list[str] = []
    texts: list[str] = []
    owners: list[int] = []
    indices_by_owner: list[list[int]] = [[] for _ in records]
    seen_ids: set[str] = set()
    for owner, record in enumerate(records):
        manifest_record = manifest_by_pair.get(record.pair_id)
        if manifest_record is None:
            continue
        for negative in manifest_record.negatives:
            if negative.negative_id in seen_ids:
                raise ValueError(f"duplicate q- ID: {negative.negative_id}")
            seen_ids.add(negative.negative_id)
            negative_index = len(ids)
            ids.append(negative.negative_id)
            texts.append(negative.text)
            owners.append(owner)
            indices_by_owner[owner].append(negative_index)
    if not ids:
        raise ValueError("q- manifest contains no usable negative queries")
    return NegativeQueryLayout(
        ids=ids,
        texts=texts,
        owners=np.asarray(owners, dtype=np.int64),
        indices_by_owner=indices_by_owner,
    )


def _normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def load_image_map(path: Path, image_root: Path) -> tuple[list[str], list[Path]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"invalid CIRR image split: {path}")
    root = image_root.resolve()
    resolved: dict[str, Path] = {}
    for image_id, mapped in payload.items():
        if not isinstance(image_id, str) or not isinstance(mapped, str):
            raise TypeError(f"invalid CIRR image split entry: {image_id!r}")
        relative = Path(mapped.removeprefix("./"))
        candidate = (root / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not candidate.is_relative_to(root):
            raise ValueError(f"unsafe CIRR image path: {mapped}")
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        resolved[image_id] = candidate
    image_ids = sorted(resolved)
    return image_ids, [resolved[image_id] for image_id in image_ids]


def load_cirr_positive_records(
    annotation_path: Path,
    image_ids: list[str],
    image_paths: list[Path],
) -> list[CIRRPositiveRecord]:
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(annotations, list):
        raise TypeError(f"invalid CIRR annotation file: {annotation_path}")
    path_by_id = dict(zip(image_ids, image_paths, strict=True))
    records: list[CIRRPositiveRecord] = []
    seen_pairs: set[int] = set()
    for row_number, row in enumerate(annotations):
        if not isinstance(row, dict):
            raise TypeError(f"CIRR annotation row {row_number} is not an object")
        pair_id = row.get("pairid")
        reference_id = row.get("reference")
        target_id = row.get("target_hard", row.get("target"))
        text = row.get("caption")
        if not isinstance(pair_id, int) or isinstance(pair_id, bool):
            raise ValueError(f"CIRR annotation row {row_number} has invalid pairid")
        if pair_id in seen_pairs:
            raise ValueError(f"duplicate CIRR pairid: {pair_id}")
        if not isinstance(reference_id, str) or reference_id not in path_by_id:
            raise ValueError(f"pair {pair_id} has unknown reference image")
        if not isinstance(target_id, str) or target_id not in path_by_id:
            raise ValueError(f"pair {pair_id} has unknown target image")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"pair {pair_id} has no modification text")
        seen_pairs.add(pair_id)
        records.append(
            CIRRPositiveRecord(
                pair_id=pair_id,
                reference_id=reference_id,
                reference_path=path_by_id[reference_id],
                target_id=target_id,
                target_path=path_by_id[target_id],
                text=text.strip(),
            )
        )
    if not records:
        raise ValueError("CIRR train annotations are empty")
    return records


class OnPolicyTrainingDataset(Dataset):
    def __init__(
        self,
        config: OnPolicyDistillationConfig,
        records: list[CIRRPositiveRecord],
        image_ids: list[str],
        image_paths: list[Path],
        mining_dir: Path,
        *,
        expected_epoch: int,
    ) -> None:
        self.records = records
        if len(image_ids) != len(image_paths):
            raise ValueError("CIRR image IDs and paths have different lengths")
        self.image_paths = image_paths
        image_index = {image_id: index for index, image_id in enumerate(image_ids)}
        self.reference_indices = np.asarray(
            [image_index[record.reference_id] for record in records], dtype=np.int64
        )
        metadata = json.loads((mining_dir / "metadata.json").read_text(encoding="utf-8"))
        if not metadata.get("complete"):
            raise RuntimeError(f"on-policy mining cache is incomplete: {mining_dir}")
        if int(metadata.get("epoch", -1)) != expected_epoch:
            raise ValueError("on-policy mining cache belongs to a different epoch")
        if int(metadata.get("sample_count", -1)) != len(records):
            raise ValueError("on-policy mining sample count does not match CIRR train")
        if int(metadata.get("top_k", -1)) != config.mining.top_k:
            raise ValueError("on-policy mining Top-K does not match config")
        self.candidate_indices = np.load(
            mining_dir / "candidate_indices.npy", mmap_mode="r"
        )
        target_indices = np.load(mining_dir / "target_indices.npy", mmap_mode="r")
        if metadata.get("target_forced_into_candidates", True) != config.mining.force_target_insertion:
            raise ValueError("target insertion mode does not match mining cache")
        self.teacher_filter_target_ranks = None
        self.target_positions = None
        if not config.mining.force_target_insertion:
            self.teacher_filter_target_ranks = np.load(
                mining_dir / "teacher_filter_target_ranks.npy", mmap_mode="r"
            )
            if self.teacher_filter_target_ranks.shape != (len(records),):
                raise ValueError("invalid teacher filter rank cache")
            matches = self.candidate_indices == target_indices[:, None]
            self.target_positions = np.where(matches.any(axis=1), matches.argmax(axis=1), -1)
        self.uses_teacher = config.objective.needs_teacher_scores
        self.teacher_scores = (
            np.load(mining_dir / "teacher_scores.npy", mmap_mode="r")
            if self.uses_teacher
            else None
        )
        self.teacher_target_top1 = (
            np.load(mining_dir / "teacher_target_top1.npy", mmap_mode="r")
            if self.uses_teacher
            else None
        )
        self.raw_target_in_topk = np.load(
            mining_dir / "raw_target_in_topk.npy", mmap_mode="r"
        )
        self.negative_query_layout = None
        self.negative_query_candidate_indices = None
        if config.mining.include_negative_queries_as_candidates:
            if config.data.negative_query_manifest is None:
                raise ValueError("mixed q- candidates require a manifest")
            self.negative_query_layout = load_negative_query_layout(
                config.data.negative_query_manifest,
                records,
            )
            self.negative_query_candidate_indices = np.load(
                mining_dir / "negative_query_candidate_indices.npy",
                mmap_mode="r",
            )
        expected = (len(records), config.mining.top_k)
        if self.candidate_indices.shape != expected:
            raise ValueError("on-policy candidate cache has an invalid shape")
        if self.teacher_scores is not None and self.teacher_scores.shape != expected:
            raise ValueError("Teacher candidate score cache has an invalid shape")
        if self.negative_query_candidate_indices is not None:
            if self.negative_query_candidate_indices.shape != expected:
                raise ValueError("q- candidate cache has an invalid shape")
            image_slots = self.candidate_indices >= 0
            negative_slots = self.negative_query_candidate_indices >= 0
            if not bool((image_slots ^ negative_slots).all()):
                raise ValueError(
                    "every mixed candidate slot must contain exactly one embedding"
                )
            if bool(
                (
                    self.negative_query_candidate_indices[negative_slots]
                    >= len(self.negative_query_layout.ids)
                ).any()
            ):
                raise ValueError("q- candidate index is out of range")
        if target_indices.shape != (len(records),) or (
            config.mining.force_target_insertion and not np.array_equal(
                self.candidate_indices[:, 0], target_indices
            )
        ):
            raise ValueError(
                "on-policy cache does not contain every target at candidate index zero"
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        example = {
            "pair_id": record.pair_id,
            "reference_index": int(self.reference_indices[index]),
            "candidate_indices": np.asarray(
                self.candidate_indices[index], dtype=np.int64
            ).copy(),
            "text": record.text,
            "raw_target_in_topk": bool(self.raw_target_in_topk[index]),
        }
        if self.teacher_scores is not None and self.teacher_target_top1 is not None:
            example["teacher_scores"] = np.asarray(
                self.teacher_scores[index], dtype=np.float32
            )
            example["teacher_target_top1"] = bool(
                self.teacher_target_top1[index]
            )
        if self.teacher_filter_target_ranks is not None:
            example["teacher_filter_target_rank"] = int(self.teacher_filter_target_ranks[index])
            example["target_position"] = int(self.target_positions[index])
        if (
            self.negative_query_layout is not None
            and self.negative_query_candidate_indices is not None
        ):
            negative_indices = np.asarray(
                self.negative_query_candidate_indices[index], dtype=np.int64
            )
            example["negative_query_candidate_texts"] = [
                (
                    None
                    if negative_index < 0
                    else self.negative_query_layout.texts[int(negative_index)]
                )
                for negative_index in negative_indices
            ]
        return example


class InMemoryReservoirDataset(Dataset):
    """Lightweight index-only dataset for live Top-M candidate screening."""

    def __init__(
        self,
        records: list[CIRRPositiveRecord],
        image_ids: list[str],
        candidate_indices: np.ndarray,
        anchor_scores: np.ndarray | None,
        target_indices: np.ndarray,
        raw_target_in_reservoir: np.ndarray,
        teacher_scores: np.ndarray | None = None,
    ) -> None:
        self.records = records
        image_index = {image_id: index for index, image_id in enumerate(image_ids)}
        self.reference_indices = np.asarray(
            [image_index[record.reference_id] for record in records], dtype=np.int64
        )
        self.candidate_indices = candidate_indices
        self.anchor_scores = anchor_scores
        self.teacher_scores = teacher_scores
        if teacher_scores is not None and teacher_scores.shape != candidate_indices.shape:
            raise ValueError("Teacher scores do not match in-memory candidates")
        self.target_indices = target_indices
        self.raw_target_in_reservoir = raw_target_in_reservoir
        if candidate_indices.ndim != 2 or candidate_indices.shape[0] != len(records):
            raise ValueError("in-memory reservoir candidates have an invalid shape")
        if anchor_scores is not None and anchor_scores.shape != candidate_indices.shape:
            raise ValueError("anchor scores do not match in-memory candidates")
        if target_indices.shape != (len(records),):
            raise ValueError("in-memory target indices have an invalid shape")
        if raw_target_in_reservoir.shape != (len(records),):
            raise ValueError("in-memory target recall mask has an invalid shape")
        if not np.array_equal(candidate_indices[:, 0], target_indices):
            raise ValueError("every in-memory reservoir must put target at index zero")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        example = {
            "pair_id": record.pair_id,
            "reference_index": int(self.reference_indices[index]),
            "target_index": int(self.target_indices[index]),
            "reservoir_candidate_indices": np.asarray(
                self.candidate_indices[index], dtype=np.int64
            ).copy(),
            "raw_target_in_reservoir": bool(
                self.raw_target_in_reservoir[index]
            ),
            "text": record.text,
        }
        if self.anchor_scores is not None:
            example["reservoir_anchor_scores"] = np.asarray(
                self.anchor_scores[index], dtype=np.float32
            ).copy()
        if self.teacher_scores is not None:
            example["reservoir_teacher_scores"] = np.asarray(
                self.teacher_scores[index], dtype=np.float32
            ).copy()
        return example


class InMemoryReservoirCollator:
    """Tokenize queries while leaving all image decoding to the live screener."""

    def __init__(self, processor) -> None:
        self.processor = processor

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        text = self.processor(
            text=[example["text"] for example in examples],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        batch = {
            "pair_ids": [example["pair_id"] for example in examples],
            "reference_indices": torch.tensor(
                [example["reference_index"] for example in examples],
                dtype=torch.long,
            ),
            "target_indices": torch.tensor(
                [example["target_index"] for example in examples],
                dtype=torch.long,
            ),
            "reservoir_candidate_indices": torch.from_numpy(
                np.stack(
                    [
                        example["reservoir_candidate_indices"]
                        for example in examples
                    ]
                )
            ),
            "raw_target_in_reservoir": torch.tensor(
                [example["raw_target_in_reservoir"] for example in examples],
                dtype=torch.bool,
            ),
            "input_ids": text["input_ids"],
            "attention_mask": text.get("attention_mask"),
        }
        if "reservoir_teacher_scores" in examples[0]:
            if any("reservoir_teacher_scores" not in example for example in examples):
                raise ValueError("batch mixes scored and unscored reservoirs")
            batch["reservoir_teacher_scores"] = torch.from_numpy(
                np.stack([example["reservoir_teacher_scores"] for example in examples])
            )
        if "reservoir_anchor_scores" in examples[0]:
            if any("reservoir_anchor_scores" not in example for example in examples):
                raise ValueError("batch mixes anchored and unanchored reservoirs")
            batch["reservoir_anchor_scores"] = torch.from_numpy(
                np.stack(
                    [example["reservoir_anchor_scores"] for example in examples]
                )
            )
        return batch


class OnPolicyTrainingCollator:
    def __init__(self, processor, image_paths: list[Path]) -> None:
        self.processor = processor
        self.image_paths = image_paths

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        unique_indices: list[int] = []
        unique_position: dict[int, int] = {}

        def position(image_index: int) -> int:
            if image_index < 0 or image_index >= len(self.image_paths):
                raise IndexError(f"CIRR candidate image index out of range: {image_index}")
            if image_index not in unique_position:
                unique_position[image_index] = len(unique_indices)
                unique_indices.append(image_index)
            return unique_position[image_index]

        reference_positions = [
            position(int(example["reference_index"])) for example in examples
        ]
        candidate_positions: list[list[int]] = []
        candidate_is_negative_query: list[list[bool]] = []
        negative_query_texts: list[str] = []
        negative_query_owners: list[int] = []
        negative_query_positions: list[int] = []
        for owner, example in enumerate(examples):
            negative_texts = example.get("negative_query_candidate_texts")
            if negative_texts is None:
                negative_texts = [None] * len(example["candidate_indices"])
            if len(negative_texts) != len(example["candidate_indices"]):
                raise ValueError("q- candidate texts do not match candidate slots")
            row_positions: list[int] = []
            row_is_negative: list[bool] = []
            for candidate_position, (image_index, negative_text) in enumerate(
                zip(example["candidate_indices"], negative_texts, strict=True)
            ):
                is_negative = negative_text is not None
                row_is_negative.append(is_negative)
                if is_negative:
                    row_positions.append(reference_positions[owner])
                    negative_query_texts.append(negative_text)
                    negative_query_owners.append(owner)
                    negative_query_positions.append(candidate_position)
                else:
                    row_positions.append(position(int(image_index)))
            candidate_positions.append(row_positions)
            candidate_is_negative_query.append(row_is_negative)
        candidate_count = len(candidate_positions[0])
        if any(len(row) != candidate_count for row in candidate_positions):
            raise ValueError("on-policy candidate rows have inconsistent lengths")
        images = [_load_image(self.image_paths[index]) for index in unique_indices]
        image_inputs = self.processor(images=images, return_tensors="pt")
        text = self.processor(
            text=[example["text"] for example in examples],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        negative_text = (
            self.processor(
                text=negative_query_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            if negative_query_texts
            else None
        )
        batch = {
            "pair_ids": [example["pair_id"] for example in examples],
            "unique_pixel_values": image_inputs["pixel_values"],
            "reference_image_positions": torch.tensor(
                reference_positions, dtype=torch.long
            ),
            "candidate_image_positions": torch.tensor(
                candidate_positions, dtype=torch.long
            ),
            "candidate_is_negative_query": torch.tensor(
                candidate_is_negative_query, dtype=torch.bool
            ),
            "unique_image_count": torch.tensor(len(unique_indices), dtype=torch.long),
            "input_ids": text["input_ids"],
            "attention_mask": text.get("attention_mask"),
            "negative_query_input_ids": (
                None if negative_text is None else negative_text["input_ids"]
            ),
            "negative_query_attention_mask": (
                None
                if negative_text is None
                else negative_text.get("attention_mask")
            ),
            "negative_query_owners": torch.tensor(
                negative_query_owners, dtype=torch.long
            ),
            "negative_query_positions": torch.tensor(
                negative_query_positions, dtype=torch.long
            ),
            "raw_target_in_topk": torch.tensor(
                [example["raw_target_in_topk"] for example in examples],
                dtype=torch.bool,
            ),
        }
        if "teacher_scores" in examples[0]:
            if any("teacher_scores" not in example for example in examples):
                raise ValueError("batch mixes Teacher and non-Teacher candidate rows")
            batch["teacher_scores"] = torch.from_numpy(
                np.stack([example["teacher_scores"] for example in examples])
            )
            batch["teacher_target_top1"] = torch.tensor(
                [example["teacher_target_top1"] for example in examples],
                dtype=torch.bool,
            )
        if "teacher_filter_target_rank" in examples[0]:
            for key in ("teacher_filter_target_rank", "target_position"):
                batch[key] = torch.tensor([example[key] for example in examples], dtype=torch.long)
        return batch


def _load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")
