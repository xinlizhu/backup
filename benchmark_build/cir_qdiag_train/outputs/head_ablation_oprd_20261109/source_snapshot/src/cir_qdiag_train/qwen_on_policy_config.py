from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from cir_qdiag_train.config import EvalConfig, StrictConfig
from cir_qdiag_train.on_policy_config import (
    OnPolicyExperimentConfig,
    OnPolicyMiningConfig,
    OnPolicyObjectiveConfig,
    OnPolicyTeacherConfig,
)


class QwenOnPolicyDataConfig(StrictConfig):
    num_workers: int = Field(default=0, ge=0)
    prefetch_factor: int = Field(default=2, ge=1)
    train_annotation_path: Path
    train_image_split_path: Path
    val_annotation_path: Path
    val_image_split_path: Path
    image_root: Path
    teacher_cache_dir: Path
    student_gallery_cache_dir: Path | None = None


class QwenQueryLoRAConfig(StrictConfig):
    variant: Literal["qwen3_vl_embedding_query_lora", "qwen3_vl_embedding_shared_lora", "qwen3_vl_instruct_query_lora", "qwen3_vl_shared_lora", "qwen3_5_shared_lora", "qwen3_5_full", "llava_shared_lora"] = (
        "qwen3_vl_embedding_query_lora"
    )
    pretrained_model_name_or_path: str
    local_files_only: bool = True
    trust_remote_code: bool = True
    lora_rank: int = Field(default=16, ge=1)
    lora_alpha: float = Field(default=32.0, gt=0.0)
    lora_dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    lora_target_modules: Literal["attention", "all-linear"] = "attention"
    gradient_checkpointing: bool = True
    max_length: int = Field(default=1024, ge=64)
    min_pixels: int = Field(default=4096, ge=1024)
    max_pixels: int = Field(default=262144, ge=4096)
    llava_image_size: int = Field(default=512, ge=16)

    @model_validator(mode="after")
    def validate_pixels(self) -> "QwenQueryLoRAConfig":
        if self.max_pixels < self.min_pixels:
            raise ValueError("max_pixels must be at least min_pixels")
        if self.shared_lora and not self.llava and self.lora_target_modules != "all-linear":
            raise ValueError("shared Qwen LoRA requires lora_target_modules: all-linear")
        if self.variant == "qwen3_vl_instruct_query_lora" and self.lora_target_modules != "attention":
            raise ValueError("Instruct query-only control uses the original attention LoRA targets")
        return self

    @property
    def shared_lora(self) -> bool:
        return self.variant in {"qwen3_vl_embedding_shared_lora", "qwen3_vl_shared_lora", "qwen3_5_shared_lora", "llava_shared_lora"}

    @property
    def llava(self) -> bool:
        return self.variant == "llava_shared_lora"

    @property
    def qwen35(self) -> bool:
        return self.variant in {"qwen3_5_shared_lora", "qwen3_5_full"}

    @property
    def train_gallery(self) -> bool:
        return self.shared_lora or self.variant == "qwen3_5_full"

    @property
    def native_retrieval(self) -> bool:
        return self.qwen35 or self.llava or self.variant in {"qwen3_vl_embedding_shared_lora", "qwen3_vl_instruct_query_lora", "qwen3_vl_shared_lora"}


class QwenOnPolicyOptimizerConfig(StrictConfig):
    epochs: int = Field(default=8, ge=1)
    world_size: int = Field(default=1, ge=1)
    batch_size_per_gpu: int = Field(default=8, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    learning_rate: float = Field(default=0.00002, gt=0.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    warmup_ratio: float = Field(default=0.05, ge=0.0, lt=1.0)
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    mixed_precision: Literal["no", "bf16", "fp16"] = "bf16"
    log_every_steps: int = Field(default=20, ge=1)
    save_every_epochs: int = Field(default=2, ge=1)
    candidate_encode_batch_size: int = Field(default=2, ge=1)
    candidate_gradient_cache: bool = True


class QwenOnPolicyDistillationConfig(StrictConfig):
    experiment: OnPolicyExperimentConfig
    data: QwenOnPolicyDataConfig
    model: QwenQueryLoRAConfig
    objective: OnPolicyObjectiveConfig
    teacher: OnPolicyTeacherConfig = Field(default_factory=OnPolicyTeacherConfig)
    mining: OnPolicyMiningConfig = Field(default_factory=OnPolicyMiningConfig)
    optimizer: QwenOnPolicyOptimizerConfig = Field(
        default_factory=QwenOnPolicyOptimizerConfig
    )
    evaluation: EvalConfig = Field(default_factory=EvalConfig)

    @model_validator(mode="after")
    def validate_experiment(self) -> "QwenOnPolicyDistillationConfig":
        if self.optimizer.world_size > 1:
            if not (self.model.llava or self.model.variant in {"qwen3_vl_embedding_shared_lora", "qwen3_vl_shared_lora"}):
                raise ValueError("This multi-GPU entry point supports LLaVA and shared Qwen Embedding/Instruct")
            if self.optimizer.gradient_accumulation_steps != 1:
                raise ValueError("LLaVA synchronous multi-GPU currently requires accumulation=1")
            if self.objective.uses_teacher_filter:
                raise ValueError("LLaVA synchronous multi-GPU uses unfiltered queries")
        if not self.model.train_gallery and self.data.student_gallery_cache_dir is None:
            raise ValueError("legacy query-only Qwen requires a fixed student gallery cache")
        if self.model.qwen35 and self.data.student_gallery_cache_dir is not None:
            raise ValueError("Qwen3.5 updates both sides and must not use a fixed student gallery")
        if self.model.llava and self.data.student_gallery_cache_dir is not None:
            raise ValueError("LLaVA updates both sides and must not use a fixed student gallery")
        if self.model.variant == "qwen3_vl_embedding_shared_lora" and self.data.student_gallery_cache_dir is not None:
            raise ValueError("Shared Embedding updates both sides; fixed student gallery is forbidden")
        if self.teacher.backend != "embedding_cache":
            raise ValueError("Qwen training requires an embedding-cache Teacher")
        if self.objective.loss_type not in {"rank_kd", "candidate_infonce", "centered_mse"}:
            raise ValueError("Qwen supports Rank KD, candidate InfoNCE and centered MSE")
        if self.objective.centered_response_weight != 0:
            raise ValueError("Qwen query-LoRA OPD is a Rank-only experiment")
        if self.mining.candidate_source not in {"student", "random"}:
            raise ValueError("Qwen supports Student or random candidates")
        if self.mining.live_reservoir_selection or self.mining.reservoir_top_k is not None:
            raise ValueError("Qwen does not support live reservoirs")
        if not self.mining.force_target_insertion:
            raise ValueError("Qwen requires target at candidate zero")
        if self.mining.refresh_mode != "every_epoch":
            raise ValueError("Qwen OPD refreshes Student candidates every epoch")
        if self.mining.refresh_every_epochs != 1:
            raise ValueError("Qwen OPD supports one candidate refresh per epoch")
        if self.optimizer.save_every_epochs > self.optimizer.epochs:
            raise ValueError("save_every_epochs cannot exceed total epochs")
        return self

    def finalize_paths(self, config_path: Path) -> "QwenOnPolicyDistillationConfig":
        project_root = config_path.resolve().parent.parent
        for owner, field_name in (
            (self.experiment, "output_dir"),
            (self.data, "train_annotation_path"),
            (self.data, "train_image_split_path"),
            (self.data, "val_annotation_path"),
            (self.data, "val_image_split_path"),
            (self.data, "image_root"),
            (self.data, "teacher_cache_dir"),
            (self.data, "student_gallery_cache_dir"),
        ):
            value = getattr(owner, field_name)
            if value is None:
                continue
            if not value.is_absolute():
                setattr(owner, field_name, (project_root / value).resolve())
        local_model = Path(self.model.pretrained_model_name_or_path)
        local_candidate = (project_root / local_model).resolve()
        if not local_model.is_absolute() and local_candidate.exists():
            self.model.pretrained_model_name_or_path = str(local_candidate)
        return self


def load_qwen_on_policy_config(
    path: str | Path,
) -> QwenOnPolicyDistillationConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return QwenOnPolicyDistillationConfig.model_validate(payload).finalize_paths(
        config_path
    )
