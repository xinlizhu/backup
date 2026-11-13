from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(ConfigModel):
    annotation_path: Path
    image_split_path: Path
    image_root: Path
    split: Literal["train"] = "train"
    verify_images: bool = True


class SamplingConfig(ConfigModel):
    limit: int | None = Field(default=50, gt=0)
    seed: int = 42
    strategy: Literal["random", "first"] = "random"


class ModelConfig(ConfigModel):
    backend: Literal["openai"] = "openai"
    api_base: str = "${OPENAI_BASE_URL}"
    api_key: str = "${OPENAI_API_KEY}"
    organization: str | None = None
    project: str | None = None
    timeout_seconds: float = Field(default=600.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    image_detail: Literal["low", "high", "original", "auto"] = "high"
    structured_output_mode: Literal["json_schema", "prompt"] = "prompt"
    generation_model: str = "gpt-5.6-terra"
    validation_model: str = "gpt-5.6-terra"
    generation_reasoning_effort: Literal[
        "none", "low", "medium", "high", "xhigh", "max"
    ] | None = None
    validation_reasoning_effort: Literal[
        "none", "low", "medium", "high", "xhigh", "max"
    ] | None = None


class GenerationConfig(ConfigModel):
    max_output_tokens: int = Field(default=2048, gt=0)


class ValidationConfig(ConfigModel):
    max_output_tokens: int = Field(default=4096, gt=0)


class ParaphraseConfig(ConfigModel):
    num_variants: int = Field(default=5, ge=1, le=20)
    generation_max_output_tokens: int = Field(default=2048, gt=0)
    review_max_output_tokens: int = Field(default=4096, gt=0)
    max_workers: int = Field(default=2, ge=1)


class RuntimeConfig(ConfigModel):
    output_dir: Path
    cache_dir: Path | None = None
    prompt_dir: Path | None = None
    prompt_version: Literal["v8"] = "v8"
    resume: bool = True
    force_stage: Literal["prepare", "generate", "validate"] | None = None
    max_workers: int = Field(default=2, ge=1)


class PipelineConfig(ConfigModel):
    data: DataConfig
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    model: ModelConfig
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    paraphrase: ParaphraseConfig = Field(default_factory=ParaphraseConfig)
    runtime: RuntimeConfig

    def finalize_paths(self, config_path: Path) -> "PipelineConfig":
        project_root = config_path.resolve().parent.parent
        for field_name in ("annotation_path", "image_split_path", "image_root"):
            value = getattr(self.data, field_name)
            if not value.is_absolute():
                setattr(self.data, field_name, (project_root / value).resolve())
        if self.runtime.prompt_dir is None:
            self.runtime.prompt_dir = project_root / "prompts"
        elif not self.runtime.prompt_dir.is_absolute():
            self.runtime.prompt_dir = (project_root / self.runtime.prompt_dir).resolve()
        if not self.runtime.output_dir.is_absolute():
            self.runtime.output_dir = (project_root / self.runtime.output_dir).resolve()
        if self.runtime.cache_dir is None:
            self.runtime.cache_dir = self.runtime.output_dir / "cache"
        elif not self.runtime.cache_dir.is_absolute():
            self.runtime.cache_dir = (project_root / self.runtime.cache_dir).resolve()
        return self


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> PipelineConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if overrides:
        _deep_update(raw, overrides)
    return PipelineConfig.model_validate(raw).finalize_paths(config_path)


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
