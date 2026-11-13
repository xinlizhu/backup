"""Ordered, bounded CPU preprocessing for gallery and query inference."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def inference_loader_settings(
    *,
    default_workers: int = 8,
    default_prefetch: int = 2,
    num_workers: int | None = None,
    prefetch_factor: int | None = None,
) -> tuple[int, int]:
    """Explicit CLI options > inference-only environment > config defaults."""
    workers = int(
        os.environ.get("CIR_INFERENCE_NUM_WORKERS", default_workers)
        if num_workers is None else num_workers
    )
    prefetch = int(
        os.environ.get("CIR_INFERENCE_PREFETCH_FACTOR", default_prefetch)
        if prefetch_factor is None else prefetch_factor
    )
    if workers < 0 or prefetch < 1:
        raise ValueError("inference num_workers must be >= 0 and prefetch_factor >= 1")
    return workers, prefetch


class _InferenceDataset(Dataset):
    def __init__(self, paths: Sequence[Path], texts: Sequence[str] | None, start: int):
        self.paths = [str(path) for path in paths]
        self.texts = None if texts is None else list(texts)
        self.start = start

    def __len__(self):
        return len(self.paths) - self.start

    def __getitem__(self, index):
        index += self.start
        return index, self.paths[index], None if self.texts is None else self.texts[index]


class _InferenceCollator:
    def __init__(self, processor):
        # Only CPU preprocessing state goes to workers, never the model/CUDA state.
        self.processor = processor

    def __call__(self, examples):
        images = []
        try:
            for _, path, _ in examples:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            result = {
                "indices": torch.tensor([row[0] for row in examples], dtype=torch.long),
                "pixel_values": self.processor(images=images, return_tensors="pt")[
                    "pixel_values"
                ].contiguous(),
            }
        finally:
            for image in images:
                image.close()
        if examples[0][2] is not None:
            text_inputs = self.processor(
                text=[row[2] for row in examples],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            result["input_ids"] = text_inputs["input_ids"]
            if text_inputs.get("attention_mask") is not None:
                result["attention_mask"] = text_inputs["attention_mask"]
        return result


def _initialize_inference_worker(worker_id):
    del worker_id
    torch.set_num_threads(1)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def build_inference_loader(
    *,
    image_paths: Sequence[Path],
    processor,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
    texts: Sequence[str] | None = None,
    start: int = 0,
) -> DataLoader:
    """Decode AND preprocess ahead of GPU inference without changing batch order.

    Spawn is safe even when the parent has already initialized CUDA. A private
    generator keeps loader creation from consuming the training RNG during mining.
    No persistent workers/cache: each one-pass loader releases its resources.
    """
    if batch_size < 1 or num_workers < 0 or prefetch_factor < 1:
        raise ValueError("invalid inference loader settings")
    if not 0 <= start <= len(image_paths):
        raise ValueError("invalid inference loader start index")
    if texts is not None and len(texts) != len(image_paths):
        raise ValueError("inference images and texts must have equal lengths")
    kwargs = {
        "dataset": _InferenceDataset(image_paths, texts, start),
        "batch_size": batch_size,
        "shuffle": False,
        "drop_last": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": _InferenceCollator(processor),
        "generator": torch.Generator().manual_seed(0),
    }
    if num_workers:
        kwargs.update(
            prefetch_factor=prefetch_factor,
            multiprocessing_context="spawn",
            worker_init_fn=_initialize_inference_worker,
        )
    print(
        f"[inference loader] images={len(image_paths) - start} batch_size={batch_size} "
        f"num_workers={num_workers} prefetch_factor={prefetch_factor if num_workers else 0} "
        f"pin_memory={pin_memory} start={start}",
        flush=True,
    )
    return DataLoader(**kwargs)


def inference_tensor_to_device(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.to(device, non_blocking=device.type == "cuda" and value.is_pinned())
