"""Qwen3-VL-Embedding adapter for the CIR-QDiag extraction interface."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


QWEN_MODEL_NAME = "Qwen3-VL-Embedding-2B"
QUERY_INSTRUCTION = "Represent the composed image retrieval query."
TARGET_INSTRUCTION = "Represent the image for retrieval."
IMAGE_FACTOR = 32
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 256 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_LENGTH = 8192


def build_query_inputs(records: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    """Convert diagnostic query rows into the Qwen embedder's native payloads."""

    return [
        {
            "image": str(record["query_image_signature"]),
            "text": str(record["instruction"]),
            "instruction": QUERY_INSTRUCTION,
        }
        for record in records
    ]


def build_reference_query_inputs(image_paths: Sequence[Path]) -> list[dict[str, str]]:
    """Build image-only inputs while preserving the composed-query role."""

    return [
        {
            "image": str(image_path),
            "instruction": QUERY_INSTRUCTION,
        }
        for image_path in image_paths
    ]


def build_target_inputs(image_paths: Sequence[Path]) -> list[dict[str, str]]:
    """Convert target image paths into native image-retrieval payloads."""

    return [
        {
            "image": str(image_path),
            "instruction": TARGET_INSTRUCTION,
        }
        for image_path in image_paths
    ]


def protocol_metadata() -> dict[str, Any]:
    return {
        "pooling": "last_non_padding_token",
        "normalization": "l2",
        "query_content": ["reference_image", "modification_text"],
        "query_instruction": QUERY_INSTRUCTION,
        "target_content": ["target_image"],
        "target_instruction": TARGET_INSTRUCTION,
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
        "max_length": MAX_LENGTH,
    }


def reference_protocol_metadata() -> dict[str, Any]:
    return {
        "role": "query",
        "content": ["reference_image"],
        "instruction": QUERY_INSTRUCTION,
    }


def _torch_dtype(dtype_arg: str, device: Any) -> Any:
    import torch

    if dtype_arg == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return mapping[dtype_arg]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype for Qwen3-VL: {dtype_arg}") from exc


class Qwen3VLDiagnosticAdapter:
    """Expose Qwen3VLEmbedder through the methods used by extraction."""

    def __init__(self, model_path: Path, device: Any, dtype: str) -> None:
        import torch

        self.model_path = Path(model_path)
        self.device = device
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"Qwen3-VL model path does not exist: {self.model_path}")

        model_root = str(self.model_path)
        if model_root not in sys.path:
            sys.path.insert(0, model_root)
        from scripts.qwen3_vl_embedding import Qwen3VLEmbedder  # type: ignore

        self.embedder = Qwen3VLEmbedder(
            model_name_or_path=model_root,
            max_length=MAX_LENGTH,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS,
            default_instruction="Represent the user's input.",
            device=str(device),
            torch_dtype=_torch_dtype(dtype, device),
            local_files_only=True,
        )
        self.embedder.model.eval()
        self._torch = torch

    def _encode(self, inputs: Sequence[dict[str, str]]) -> np.ndarray:
        if not inputs:
            hidden_size = int(self.embedder.model.config.text_config.hidden_size)
            return np.empty((0, hidden_size), dtype=np.float32)
        with self._torch.inference_mode():
            embeddings = self.embedder.process(list(inputs), normalize=True)
        return embeddings.detach().float().cpu().numpy().astype(np.float32, copy=False)

    def encode_query_batch(
        self,
        rows: Any,
        image_cache: Path,
        preloaded_images: Any = None,
    ) -> np.ndarray:
        del image_cache, preloaded_images
        return self._encode(build_query_inputs(rows.to_dict(orient="records")))

    def encode_reference_query_batch(self, images: Sequence[Path]) -> np.ndarray:
        return self._encode(build_reference_query_inputs([Path(image) for image in images]))

    def encode_image_batch(
        self,
        images: Sequence[Path],
        identifiers: Sequence[str] | None = None,
    ) -> np.ndarray:
        del identifiers
        return self._encode(build_target_inputs([Path(image) for image in images]))
