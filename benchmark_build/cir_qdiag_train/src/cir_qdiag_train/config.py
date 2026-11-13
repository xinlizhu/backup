from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExperimentConfig(StrictConfig):
    name: str = Field(min_length=1)
    seed: int = 42
    negative_mode: Literal["none", "add_substitute", "add_substitute_delete"]
    output_dir: Path

    @property
    def use_hard_negatives(self) -> bool:
        return self.negative_mode != "none"

    @property
    def include_delete(self) -> bool:
        return self.negative_mode == "add_substitute_delete"


class DataConfig(StrictConfig):
    train_manifest: Path
    train_image_split_path: Path | None = None
    val_annotation_path: Path
    val_image_split_path: Path
    image_root: Path
    max_non_delete_negatives_per_sample: int = Field(default=6, ge=1)
    max_delete_negatives_per_sample: int = Field(default=3, ge=1)
    candidate_image_mode: Literal["none", "image_set"] = "none"
    max_candidate_images_per_sample: int = Field(default=4, ge=1)
    num_workers: int = Field(default=8, ge=0)

    @model_validator(mode="after")
    def validate_candidate_images(self) -> "DataConfig":
        if self.candidate_image_mode == "image_set" and self.train_image_split_path is None:
            raise ValueError(
                "candidate_image_mode=image_set requires train_image_split_path"
            )
        return self


class ModelConfig(StrictConfig):
    variant: Literal["clip_vit_b16", "clip_vit_l14", "bge_vl_large"]
    pretrained_model_name_or_path: str
    local_files_only: bool = False
    trust_remote_code: bool = False
    composition_head: Literal['residual_mlp', 'mmret_add', 'clip4cir_combiner', 'magiclens'] = 'residual_mlp'
    fusion_init_path: str | None = None
    hidden_dim: int = Field(default=1024, ge=1)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    freeze_backbone: bool = True
    freeze_vision_encoder: bool = False
    unfreeze_last_n_layers: int | None = Field(default=None, ge=1)
    lora_rank: int | None = Field(default=None, ge=1)
    lora_alpha: float = Field(default=32.0, gt=0.0)
    lora_dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    gradient_checkpointing: bool = False

    @model_validator(mode="after")
    def validate_finetuning(self) -> "ModelConfig":
        if self.freeze_vision_encoder and self.freeze_backbone:
            raise ValueError(
                "freeze_vision_encoder is only meaningful when freeze_backbone=false"
            )
        if self.freeze_vision_encoder and self.unfreeze_last_n_layers is not None:
            raise ValueError(
                "freeze_vision_encoder cannot be combined with last-N finetuning"
            )
        if self.freeze_vision_encoder and self.lora_rank is not None:
            raise ValueError("freeze_vision_encoder cannot be combined with LoRA")
        if self.freeze_backbone and self.unfreeze_last_n_layers is not None:
            raise ValueError(
                "unfreeze_last_n_layers requires freeze_backbone=false"
            )
        if self.lora_rank is not None:
            if not self.freeze_backbone:
                raise ValueError("LoRA requires freeze_backbone=true for the base weights")
            if self.unfreeze_last_n_layers is not None:
                raise ValueError("LoRA cannot be combined with unfreeze_last_n_layers")
        return self


class OptimizerConfig(StrictConfig):
    epochs: int = Field(default=10, ge=1)
    batch_size_per_gpu: int = Field(default=64, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0.0)
    backbone_learning_rate: float = Field(default=1e-6, gt=0.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    warmup_ratio: float = Field(default=0.05, ge=0.0, lt=1.0)
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    temperature: float = Field(default=0.07, gt=0.0)
    learnable_temperature: bool = True
    loss_mode: Literal["batch_plus_hard", "hard_only"] = "batch_plus_hard"
    hard_negative_weight: float = Field(default=1.0, ge=0.0)
    hard_negative_margin: float = Field(default=0.0, ge=0.0, le=2.0)
    candidate_negative_weight: float = Field(default=0.0, ge=0.0)
    counterfactual_square_weight: float = Field(default=0.0, ge=0.0)
    candidate_margin: float = Field(default=0.0, ge=0.0)
    counterfactual_margin: float = Field(default=0.0, ge=0.0)
    counterfactual_mining_min_advantage: float = Field(default=0.0, ge=0.0)
    mixed_precision: Literal["no", "bf16", "fp16"] = "bf16"
    log_every_steps: int = Field(default=20, ge=1)
    save_every_epochs: int = Field(default=1, ge=1)


class EvalConfig(StrictConfig):
    batch_size: int = Field(default=128, ge=1)
    num_workers: int = Field(default=8, ge=0)
    prefetch_factor: int = Field(default=2, ge=1)
    pin_memory: bool = True
    recall_k: list[int] = Field(default_factory=lambda: [1, 5, 10, 50])
    group_recall_k: list[int] = Field(default_factory=lambda: [1, 2, 3])

    @model_validator(mode="after")
    def validate_cutoffs(self) -> "EvalConfig":
        if not self.recall_k or not self.group_recall_k:
            raise ValueError("evaluation cutoff lists must be non-empty")
        if min(self.recall_k + self.group_recall_k) < 1:
            raise ValueError("evaluation cutoffs must be positive")
        return self


class TrainConfig(StrictConfig):
    experiment: ExperimentConfig
    data: DataConfig
    model: ModelConfig = Field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    evaluation: EvalConfig = Field(default_factory=EvalConfig)

    @model_validator(mode="after")
    def validate_baseline_settings(self) -> "TrainConfig":
        if self.optimizer.loss_mode == "hard_only":
            if self.experiment.negative_mode == "none":
                raise ValueError("hard_only loss requires a hard-negative mode")
            if self.optimizer.hard_negative_weight != 1.0:
                raise ValueError(
                    "hard_only loss must set hard_negative_weight=1.0; the weight is "
                    "fixed because no batch loss is mixed into the objective"
                )
        if self.experiment.negative_mode == "none" and self.optimizer.hard_negative_weight != 0:
            raise ValueError(
                "normal-training baseline must set hard_negative_weight=0 so its intent is explicit"
            )
        if (
            self.experiment.negative_mode == "none"
            and self.optimizer.hard_negative_margin != 0
        ):
            raise ValueError(
                "normal-training baseline must set hard_negative_margin=0"
            )
        if self.experiment.negative_mode != "none" and self.optimizer.hard_negative_weight <= 0:
            raise ValueError("hard-negative modes require hard_negative_weight > 0")
        uses_candidate_loss = (
            self.optimizer.candidate_negative_weight > 0
            or self.optimizer.counterfactual_square_weight > 0
        )
        if uses_candidate_loss and self.data.candidate_image_mode != "image_set":
            raise ValueError(
                "candidate losses require data.candidate_image_mode=image_set"
            )
        if (
            self.data.candidate_image_mode == "image_set"
            and not uses_candidate_loss
        ):
            raise ValueError(
                "candidate_image_mode=image_set requires a positive candidate loss weight"
            )
        if (
            self.optimizer.counterfactual_square_weight > 0
            and self.experiment.negative_mode == "none"
        ):
            raise ValueError(
                "counterfactual square loss requires typed negative queries"
            )
        return self

    def finalize_paths(self, config_path: Path) -> "TrainConfig":
        project_root = config_path.resolve().parent.parent
        for owner, field_name in (
            (self.experiment, "output_dir"),
            (self.data, "train_manifest"),
            (self.data, "val_annotation_path"),
            (self.data, "val_image_split_path"),
            (self.data, "image_root"),
        ):
            value = getattr(owner, field_name)
            if not value.is_absolute():
                setattr(owner, field_name, (project_root / value).resolve())
        if (
            self.data.train_image_split_path is not None
            and not self.data.train_image_split_path.is_absolute()
        ):
            self.data.train_image_split_path = (
                project_root / self.data.train_image_split_path
            ).resolve()
        local_model = Path(self.model.pretrained_model_name_or_path)
        local_candidate = (project_root / local_model).resolve()
        if not local_model.is_absolute() and local_candidate.exists():
            self.model.pretrained_model_name_or_path = str(local_candidate)
        return self


def load_config(path: str | Path) -> TrainConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return TrainConfig.model_validate(payload).finalize_paths(config_path)
