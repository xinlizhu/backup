from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from cir_qdiag_train.config import EvalConfig, ModelConfig, StrictConfig


class DistillationExperimentConfig(StrictConfig):
    name: str = Field(min_length=1)
    seed: int = 42
    output_dir: Path
    initialize_from: Path


class DistillationDataConfig(StrictConfig):
    train_manifest: Path
    train_image_split_path: Path
    val_annotation_path: Path
    val_image_split_path: Path
    image_root: Path
    teacher_cache_dir: Path
    student_image_cache_dir: Path
    semantic_target: Literal["object", "attribute", "predicate"] = "object"
    edit_operations: list[Literal["add", "substitute"]] = Field(
        default_factory=lambda: ["add", "substitute"]
    )
    num_workers: int = Field(default=8, ge=0)

    @model_validator(mode="after")
    def validate_operations(self) -> "DistillationDataConfig":
        if not self.edit_operations:
            raise ValueError("edit_operations must not be empty")
        if len(set(self.edit_operations)) != len(self.edit_operations):
            raise ValueError("edit_operations must not contain duplicates")
        return self


class DistillationObjectiveConfig(StrictConfig):
    positive_rank_weight: float = Field(default=0.0, ge=0.0)
    semantic_response_weight: float = Field(default=0.0, ge=0.0)
    positive_rank_temperature: float = Field(default=0.05, gt=0.0)
    semantic_response_beta: float = Field(default=0.01, gt=0.0)
    require_teacher_target_top1: bool = True
    semantic_min_teacher_target_delta: float = Field(default=0.0, ge=-2.0, le=2.0)

    @model_validator(mode="after")
    def validate_objective(self) -> "DistillationObjectiveConfig":
        if self.positive_rank_weight == 0 and self.semantic_response_weight == 0:
            raise ValueError("at least one distillation loss weight must be positive")
        return self


class DistillationOptimizerConfig(StrictConfig):
    epochs: int = Field(default=16, ge=1)
    batch_size_per_gpu: int = Field(default=128, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    learning_rate: float = Field(default=1e-5, gt=0.0)
    backbone_learning_rate: float = Field(default=5e-7, gt=0.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    warmup_ratio: float = Field(default=0.05, ge=0.0, lt=1.0)
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    temperature: float = Field(default=0.07, gt=0.0)
    learnable_temperature: bool = False
    mixed_precision: Literal["no", "bf16", "fp16"] = "bf16"
    log_every_steps: int = Field(default=20, ge=1)
    save_every_epochs: int = Field(default=4, ge=1)


class SemanticDistillationConfig(StrictConfig):
    experiment: DistillationExperimentConfig
    data: DistillationDataConfig
    model: ModelConfig
    objective: DistillationObjectiveConfig
    optimizer: DistillationOptimizerConfig
    evaluation: EvalConfig = Field(default_factory=EvalConfig)

    @model_validator(mode="after")
    def validate_training_mode(self) -> "SemanticDistillationConfig":
        if self.model.freeze_backbone:
            raise ValueError("semantic distillation must train the query/text side")
        if not self.model.freeze_vision_encoder:
            raise ValueError("semantic distillation MVP requires a frozen vision encoder")
        if self.optimizer.learnable_temperature:
            raise ValueError(
                "learnable_temperature must be false because KD uses explicit temperatures"
            )
        return self

    def finalize_paths(self, config_path: Path) -> "SemanticDistillationConfig":
        project_root = config_path.resolve().parent.parent
        for owner, field_name in (
            (self.experiment, "output_dir"),
            (self.experiment, "initialize_from"),
            (self.data, "train_manifest"),
            (self.data, "train_image_split_path"),
            (self.data, "val_annotation_path"),
            (self.data, "val_image_split_path"),
            (self.data, "image_root"),
            (self.data, "teacher_cache_dir"),
            (self.data, "student_image_cache_dir"),
        ):
            value = getattr(owner, field_name)
            if not value.is_absolute():
                setattr(owner, field_name, (project_root / value).resolve())
        local_model = Path(self.model.pretrained_model_name_or_path)
        local_candidate = (project_root / local_model).resolve()
        if not local_model.is_absolute() and local_candidate.exists():
            self.model.pretrained_model_name_or_path = str(local_candidate)
        return self


def load_distillation_config(path: str | Path) -> SemanticDistillationConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return SemanticDistillationConfig.model_validate(payload).finalize_paths(config_path)
