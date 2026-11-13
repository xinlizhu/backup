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
    split: Literal["train", "val"] = "val"


class SamplingConfig(ConfigModel):
    limit: int = Field(default=50, gt=0)
    seed: int = 42
    strategy: Literal["random", "hybrid"] = "random"
    random_count: int | None = Field(default=None, ge=0)
    targeted_count: int | None = Field(default=None, ge=0)


class ModelConfig(ConfigModel):
    backend: Literal["vllm", "transformers"] = "vllm"
    name: str = "qwen3.6-27b"
    name_or_path: Path
    api_base: str = "http://127.0.0.1:8000/v1"
    api_key: str = "EMPTY"
    model_revision: str | None = None
    device: str = "cuda"
    dtype: str = "bfloat16"
    trust_remote_code: bool = False
    timeout_seconds: float = Field(default=300.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    disable_thinking: bool = True
    mm_processor_kwargs: dict[str, Any] = Field(default_factory=lambda: {"max_pixels": 4_194_304})


class GenerationConfig(ConfigModel):
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_new_tokens: int = Field(default=2048, gt=0)


class ScreeningConfig(ConfigModel):
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_new_tokens: int = Field(default=1024, gt=0)


class SelectionConfig(ConfigModel):
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_new_tokens: int = Field(default=4096, gt=0)


class ParaphraseConfig(ConfigModel):
    num_variants: int = Field(default=5, ge=1, le=20)
    generation_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    generation_max_new_tokens: int = Field(default=8192, gt=0)
    review_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    review_max_new_tokens: int = Field(default=8192, gt=0)


class RuntimeConfig(ConfigModel):
    output_dir: Path
    cache_dir: Path | None = None
    prompt_dir: Path | None = None
    prompt_version: Literal["v5"] = "v5"
    resume: bool = True
    force_stage: Literal["prepare", "generate", "validate"] | None = None
    max_workers: int = Field(default=2, ge=1)


class PipelineConfig(ConfigModel):
    data: DataConfig
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    model: ModelConfig
    screening: ScreeningConfig = Field(default_factory=ScreeningConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    paraphrase: ParaphraseConfig = Field(default_factory=ParaphraseConfig)
    runtime: RuntimeConfig

    def finalize_paths(self, config_path: Path) -> "PipelineConfig":
        project_root = config_path.resolve().parent.parent
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
