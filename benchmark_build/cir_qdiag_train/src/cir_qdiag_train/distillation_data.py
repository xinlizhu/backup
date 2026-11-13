from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from cir_qdiag_train.distillation_config import SemanticDistillationConfig
from cir_qdiag_train.manifest import read_manifest


class SemanticDistillationDataset(Dataset):
    def __init__(self, config: SemanticDistillationConfig) -> None:
        self.config = config
        self.records = read_manifest(config.data.train_manifest)
        teacher_metadata = json.loads(
            (config.data.teacher_cache_dir / "metadata.json").read_text(encoding="utf-8")
        )
        student_metadata = json.loads(
            (config.data.student_image_cache_dir / "metadata.json").read_text(
                encoding="utf-8"
            )
        )
        if not teacher_metadata.get("complete") or not student_metadata.get("complete"):
            raise RuntimeError("distillation caches are not complete")
        if teacher_metadata.get("semantic_target") != config.data.semantic_target:
            raise ValueError("Teacher cache semantic target does not match config")
        if set(teacher_metadata.get("edit_operations", [])) != set(
            config.data.edit_operations
        ):
            raise ValueError("Teacher cache edit operations do not match config")

        signals_path = config.data.teacher_cache_dir / "teacher_signals.npz"
        with np.load(signals_path, allow_pickle=False) as loaded_signals:
            self.signals = {
                key: loaded_signals[key] for key in loaded_signals.files
            }
        sample_ids = self.signals["sample_ids"].astype(str).tolist()
        expected_sample_ids = [record.sample_id for record in self.records]
        if sample_ids != expected_sample_ids:
            raise ValueError("Teacher cache sample order does not match manifest")

        image_ids = json.loads(
            (config.data.student_image_cache_dir / "image_ids.json").read_text(
                encoding="utf-8"
            )
        )
        self.image_index = {str(value): index for index, value in enumerate(image_ids)}
        self.image_features = np.load(
            config.data.student_image_cache_dir / "image_features.npy",
            mmap_mode="r",
        )
        if self.image_features.ndim != 2 or self.image_features.shape[0] != len(image_ids):
            raise ValueError("Student image feature cache has an invalid shape")

        negative_ids = self.signals["negative_ids"].astype(str).tolist()
        negative_owner = self.signals["negative_owner"].astype(np.int64)
        self.negative_indices_by_owner: list[list[int]] = [
            [] for _ in self.records
        ]
        for index, owner in enumerate(negative_owner.tolist()):
            if owner < 0 or owner >= len(self.records):
                raise ValueError("Teacher negative owner is out of range")
            self.negative_indices_by_owner[owner].append(index)
        self.negative_text_by_id = {
            negative.negative_id: negative.text
            for record in self.records
            for negative in record.negatives
        }
        missing_negative_ids = sorted(set(negative_ids) - set(self.negative_text_by_id))
        if missing_negative_ids:
            raise ValueError(
                f"Teacher cache contains unknown negative IDs: {missing_negative_ids[:5]}"
            )
        self.negative_ids = negative_ids

        use_rank = config.objective.positive_rank_weight > 0
        self.sample_indices = [
            index
            for index in range(len(self.records))
            if use_rank or self.negative_indices_by_owner[index]
        ]
        if not self.sample_indices:
            raise ValueError("distillation dataset has no usable samples")

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        owner = self.sample_indices[item]
        record = self.records[owner]
        candidate_mask = self.signals["candidate_mask"][owner].astype(np.bool_)
        candidate_ids = self.signals["candidate_ids"][owner].astype(str)
        candidate_features = np.zeros(
            (len(candidate_ids), self.image_features.shape[1]),
            dtype=np.float32,
        )
        for candidate_index in np.nonzero(candidate_mask)[0].tolist():
            image_id = candidate_ids[candidate_index]
            try:
                feature_index = self.image_index[image_id]
            except KeyError as exc:
                raise ValueError(f"Student image cache is missing {image_id}") from exc
            candidate_features[candidate_index] = self.image_features[feature_index]

        negative_indices = self.negative_indices_by_owner[owner]
        return {
            "sample_id": record.sample_id,
            "positive_text": record.positive_text,
            "reference_feature": np.asarray(
                self.image_features[self.image_index[record.reference_id]],
                dtype=np.float32,
            ),
            "candidate_features": candidate_features,
            "candidate_mask": candidate_mask,
            "target_index": int(self.signals["target_indices"][owner]),
            "teacher_positive_scores": np.asarray(
                self.signals["positive_scores"][owner], dtype=np.float32
            ),
            "teacher_target_top1": bool(
                self.signals["teacher_target_top1"][owner]
            ),
            "negative_texts": [
                self.negative_text_by_id[self.negative_ids[index]]
                for index in negative_indices
            ],
            "teacher_negative_scores": np.asarray(
                self.signals["negative_scores"][negative_indices],
                dtype=np.float32,
            ),
        }


class SemanticDistillationCollator:
    def __init__(self, processor, *, include_negative_queries: bool) -> None:
        self.processor = processor
        self.include_negative_queries = include_negative_queries

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        positive = self.processor(
            text=[example["positive_text"] for example in examples],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        negative_texts: list[str] = []
        negative_owner: list[int] = []
        teacher_negative_scores: list[np.ndarray] = []
        if self.include_negative_queries:
            for owner, example in enumerate(examples):
                negative_texts.extend(example["negative_texts"])
                negative_owner.extend([owner] * len(example["negative_texts"]))
                teacher_negative_scores.extend(example["teacher_negative_scores"])
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
        candidate_count = len(examples[0]["candidate_mask"])
        return {
            "sample_ids": [example["sample_id"] for example in examples],
            "reference_features": torch.tensor(
                np.stack([example["reference_feature"] for example in examples]),
                dtype=torch.float32,
            ),
            "candidate_features": torch.tensor(
                np.stack([example["candidate_features"] for example in examples]),
                dtype=torch.float32,
            ),
            "candidate_mask": torch.tensor(
                np.stack([example["candidate_mask"] for example in examples]),
                dtype=torch.bool,
            ),
            "target_indices": torch.tensor(
                [example["target_index"] for example in examples], dtype=torch.long
            ),
            "teacher_positive_scores": torch.tensor(
                np.stack([example["teacher_positive_scores"] for example in examples]),
                dtype=torch.float32,
            ),
            "teacher_target_top1": torch.tensor(
                [example["teacher_target_top1"] for example in examples],
                dtype=torch.bool,
            ),
            "positive_input_ids": positive["input_ids"],
            "positive_attention_mask": positive.get("attention_mask"),
            "negative_input_ids": None if negative is None else negative["input_ids"],
            "negative_attention_mask": (
                None if negative is None else negative.get("attention_mask")
            ),
            "negative_owner": torch.tensor(negative_owner, dtype=torch.long),
            "teacher_negative_scores": (
                torch.tensor(
                    np.stack(teacher_negative_scores), dtype=torch.float32
                )
                if teacher_negative_scores
                else torch.empty((0, candidate_count), dtype=torch.float32)
            ),
        }
