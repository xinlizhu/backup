from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OAPExperimentConfig(StrictConfig):
    name: str = Field(min_length=1)
    seed: int = 42
    output_dir: Path


class OAPDataConfig(StrictConfig):
    train_manifest: Path
    val_annotation_path: Path
    val_image_split_path: Path
    image_root: Path
    max_negatives_per_semantic: int = Field(default=2, ge=1)
    num_workers: int = Field(default=8, ge=0)


class OAPModelConfig(StrictConfig):
    architecture: Literal["residual_adapter", "attention_tokens"] = (
        "residual_adapter"
    )
    variant: Literal["clip_vit_l14", "bge_vl_large"]
    pretrained_model_name_or_path: str
    base_checkpoint: Path
    local_files_only: bool = True
    trust_remote_code: bool = False
    base_hidden_dim: int = Field(default=1024, ge=1)
    base_dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    base_temperature: float = Field(default=0.07, gt=0.0)
    base_learnable_temperature: bool = True
    semantic_hidden_dim: int = Field(default=128, ge=1)
    semantic_dim: int = Field(default=64, ge=1)
    attention_hidden_dim: int = Field(default=256, ge=1)
    attention_num_heads: int = Field(default=8, ge=1)
    attention_ffn_dim: int = Field(default=512, ge=1)
    attention_dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    beta: float = Field(default=0.1, gt=0.0)

    @model_validator(mode="after")
    def validate_variant(self) -> OAPModelConfig:
        if self.variant == "bge_vl_large" and not self.trust_remote_code:
            raise ValueError("bge_vl_large requires trust_remote_code=true")
        if self.variant == "clip_vit_l14" and self.trust_remote_code:
            raise ValueError("ordinary CLIP-L must set trust_remote_code=false")
        if self.attention_hidden_dim % self.attention_num_heads:
            raise ValueError(
                "attention_hidden_dim must be divisible by attention_num_heads"
            )
        return self


class OAPOptimizerConfig(StrictConfig):
    epochs: int = Field(default=5, ge=1)
    batch_size_per_gpu: int = Field(default=64, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    warmup_ratio: float = Field(default=0.05, ge=0.0, lt=1.0)
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    ordering_temperature: float = Field(default=0.07, gt=0.0)
    retrieval_margin: float = Field(default=0.0, ge=0.0)
    patch_margin: float = Field(default=0.01, ge=0.0)
    mixed_precision: Literal["no", "bf16", "fp16"] = "bf16"
    log_every_steps: int = Field(default=50, ge=1)
    save_every_epochs: int = Field(default=2, ge=1)


class OAPEvalConfig(StrictConfig):
    batch_size: int = Field(default=128, ge=1)
    recall_k: list[int] = Field(default_factory=lambda: [1, 5, 10, 50])
    group_recall_k: list[int] = Field(default_factory=lambda: [1, 2, 3])

    @model_validator(mode="after")
    def validate_cutoffs(self) -> OAPEvalConfig:
        if not self.recall_k or not self.group_recall_k:
            raise ValueError("evaluation cutoff lists must be non-empty")
        if min(self.recall_k + self.group_recall_k) < 1:
            raise ValueError("evaluation cutoffs must be positive")
        return self


class OAPConfig(StrictConfig):
    experiment: OAPExperimentConfig
    data: OAPDataConfig
    model: OAPModelConfig
    optimizer: OAPOptimizerConfig = Field(default_factory=OAPOptimizerConfig)
    evaluation: OAPEvalConfig = Field(default_factory=OAPEvalConfig)

    @model_validator(mode="after")
    def validate_patch_margin(self) -> OAPConfig:
        # One patched component contributes beta / 3 times a cosine difference
        # in [-2, 2]. A larger margin is mathematically impossible.
        maximum = 2.0 * self.model.beta / 3.0
        if self.optimizer.patch_margin >= maximum:
            raise ValueError(
                f"patch_margin must be below 2*beta/3 ({maximum:.6g})"
            )
        return self

    def finalize_paths(self, config_path: Path) -> OAPConfig:
        project_root = config_path.resolve().parent.parent
        for owner, field_name in (
            (self.experiment, "output_dir"),
            (self.data, "train_manifest"),
            (self.data, "val_annotation_path"),
            (self.data, "val_image_split_path"),
            (self.data, "image_root"),
            (self.model, "base_checkpoint"),
        ):
            value = getattr(owner, field_name)
            if not value.is_absolute():
                setattr(owner, field_name, (project_root / value).resolve())
        local_model = Path(self.model.pretrained_model_name_or_path)
        local_candidate = (project_root / local_model).resolve()
        if not local_model.is_absolute() and local_candidate.exists():
            self.model.pretrained_model_name_or_path = str(local_candidate)
        return self


def load_oap_config(path: str | Path) -> OAPConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return OAPConfig.model_validate(payload).finalize_paths(config_path)
