from __future__ import annotations

import gc
import math
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


@dataclass
class GalleryPixelCache:
    """Process-local, preprocessed gallery pixels kept only in host memory."""

    pixel_values: torch.Tensor
    build_seconds: float
    workers: int
    build_batch_size: int
    prefetch_factor: int

    def __post_init__(self) -> None:
        if self.pixel_values.device.type != "cpu":
            raise ValueError("gallery pixel cache must remain in CPU memory")
        if self.pixel_values.ndim < 2:
            raise ValueError("gallery pixel cache has an invalid shape")

    def __len__(self) -> int:
        return int(self.pixel_values.shape[0])

    @property
    def size_bytes(self) -> int:
        return self.pixel_values.numel() * self.pixel_values.element_size()


class _GalleryPathDataset(Dataset):
    def __init__(self, image_paths: Sequence[Path]) -> None:
        self.image_paths = [str(path) for path in image_paths]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[int, str]:
        return index, self.image_paths[index]


class _GalleryPixelCollator:
    """Decode and preprocess a whole image batch inside a worker process."""

    def __init__(self, processor) -> None:
        self.processor = processor

    def __call__(
        self,
        examples: list[tuple[int, str]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.tensor([index for index, _ in examples], dtype=torch.long)
        images: list[Image.Image] = []
        try:
            for _, path in examples:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            pixel_values = self.processor(
                images=images,
                return_tensors="pt",
            )["pixel_values"]
        finally:
            for image in images:
                image.close()
        if pixel_values.shape[0] != len(examples):
            raise ValueError("processor returned the wrong gallery batch size")
        return indices, pixel_values.contiguous()


def _initialize_pixel_worker(worker_id: int) -> None:
    del worker_id
    # Each worker owns an independent image batch. Limiting intra-op threads
    # avoids multiplying a multi-process loader into severe CPU oversubscription.
    torch.set_num_threads(1)


def build_gallery_pixel_cache(
    *,
    image_paths: Sequence[Path],
    processor,
    dtype: torch.dtype,
    batch_size: int,
    workers: int,
    prefetch_factor: int,
) -> GalleryPixelCache:
    """Preprocess the immutable gallery once with a multiprocessing pipeline."""

    if not image_paths:
        raise ValueError("cannot build an empty gallery pixel cache")
    if batch_size < 1 or workers < 0 or prefetch_factor < 1:
        raise ValueError("invalid gallery pixel cache loader settings")
    started = time.perf_counter()
    loader_kwargs = {
        "dataset": _GalleryPathDataset(image_paths),
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": False,
        "drop_last": False,
        "collate_fn": _GalleryPixelCollator(processor),
    }
    if workers > 0:
        loader_kwargs.update(
            {
                "prefetch_factor": prefetch_factor,
                "persistent_workers": True,
                "multiprocessing_context": "spawn",
                "worker_init_fn": _initialize_pixel_worker,
            }
        )
    loader = DataLoader(**loader_kwargs)
    cached: torch.Tensor | None = None
    seen = torch.zeros(len(image_paths), dtype=torch.bool)
    for indices, pixels in tqdm(
        loader,
        total=math.ceil(len(image_paths) / batch_size),
        desc="Preprocess train gallery into RAM",
    ):
        if pixels.device.type != "cpu":
            raise ValueError("gallery preprocessing unexpectedly returned CUDA pixels")
        if not bool(torch.isfinite(pixels).all()):
            raise ValueError("gallery preprocessing produced non-finite pixels")
        if cached is None:
            cached = torch.empty(
                (len(image_paths), *pixels.shape[1:]),
                dtype=dtype,
                device="cpu",
            )
        elif tuple(cached.shape[1:]) != tuple(pixels.shape[1:]):
            raise ValueError("gallery processor returned inconsistent image shapes")
        cached.index_copy_(0, indices, pixels.to(dtype=dtype))
        seen[indices] = True
    if cached is None or not bool(seen.all()):
        raise RuntimeError("gallery pixel cache is incomplete")
    del loader, seen
    gc.collect()
    result = GalleryPixelCache(
        pixel_values=cached,
        build_seconds=time.perf_counter() - started,
        workers=workers,
        build_batch_size=batch_size,
        prefetch_factor=prefetch_factor,
    )
    print(
        "[gallery pixel cache] "
        f"images={len(result)} shape={tuple(result.pixel_values.shape[1:])} "
        f"dtype={result.pixel_values.dtype} "
        f"size_gib={result.size_bytes / (1024 ** 3):.3f} "
        f"workers={workers} prefetch_factor={prefetch_factor} "
        f"seconds={result.build_seconds:.2f}",
        flush=True,
    )
    return result


def iter_prefetched_pixel_chunks(
    *,
    cache: GalleryPixelCache,
    indices: Sequence[int] | torch.Tensor,
    batch_size: int,
    prefetch_batches: int,
    pin_memory: bool,
) -> Iterator[torch.Tensor]:
    """Gather CPU-cache chunks ahead of the GPU consumer in a bounded queue."""

    if batch_size < 1 or prefetch_batches < 1:
        raise ValueError("pixel-cache batch and prefetch sizes must be positive")
    index_tensor = torch.as_tensor(indices, dtype=torch.long, device="cpu")
    ranges = [
        (begin, min(begin + batch_size, len(index_tensor)))
        for begin in range(0, len(index_tensor), batch_size)
    ]
    if not ranges:
        return

    pin_enabled = [pin_memory and torch.cuda.is_available()]

    def gather(begin: int, end: int) -> torch.Tensor:
        values = cache.pixel_values.index_select(0, index_tensor[begin:end])
        if pin_enabled[0]:
            try:
                values = values.pin_memory()
            except RuntimeError as error:
                pin_enabled[0] = False
                print(
                    "[gallery pixel prefetch] pinned-memory allocation failed; "
                    f"falling back to pageable RAM: {error}",
                    flush=True,
                )
        return values

    # A single producer prevents CPU oversubscription. The expensive image
    # decoding already happened in the multiprocessing cache builder; here the
    # producer only gathers contiguous tensors and pins a few batches ahead.
    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="gallery-pixel-prefetch",
    ) as executor:
        pending: deque[Future[torch.Tensor]] = deque()
        next_range = 0
        while next_range < min(prefetch_batches, len(ranges)):
            pending.append(executor.submit(gather, *ranges[next_range]))
            next_range += 1
        while pending:
            future = pending.popleft()
            if next_range < len(ranges):
                pending.append(executor.submit(gather, *ranges[next_range]))
                next_range += 1
            yield future.result()
