from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np
import torch
from tqdm import tqdm

from cir_qdiag_train.on_policy_mining import write_json_atomic


FASHIONIQ_CATEGORIES = ("dress", "shirt", "toptee")
FASHIONIQ_RECALL_K = (1, 5, 10, 50)


@dataclass(frozen=True)
class FashionIQRecord:
    reference_id: str
    target_id: str
    captions: tuple[str, str]

    @property
    def text(self) -> str:
        return join_fashioniq_captions(self.captions)


def join_fashioniq_captions(captions: Sequence[str]) -> str:
    """Use the caption composition employed by the local FashionIQ evaluator."""

    if len(captions) != 2:
        raise ValueError(f"FashionIQ requires exactly two captions, got {len(captions)}")
    cleaned = [str(caption).strip(".?, ") for caption in captions]
    # The released val annotations contain three rows with one genuinely empty
    # caption. Preserve the standard two-caption template (" and <caption>")
    # instead of silently changing the protocol only for those rows.
    return f"{cleaned[0]} and {cleaned[1]}"


def load_fashioniq_val(
    dataset_root: str | Path,
    category: str,
) -> tuple[list[str], list[Path], list[FashionIQRecord]]:
    if category not in FASHIONIQ_CATEGORIES:
        raise ValueError(
            f"unsupported FashionIQ category {category!r}; "
            f"choose from {FASHIONIQ_CATEGORIES}"
        )
    root = Path(dataset_root).resolve()
    caption_path = root / "captions" / f"cap.{category}.val.json"
    split_path = root / "image_splits" / f"split.{category}.val.json"
    if not caption_path.is_file() or not split_path.is_file():
        raise FileNotFoundError(
            f"FashionIQ val metadata is incomplete for {category}: "
            f"{caption_path}, {split_path}"
        )
    annotations = json.loads(caption_path.read_text(encoding="utf-8"))
    gallery_ids = json.loads(split_path.read_text(encoding="utf-8"))
    if not isinstance(annotations, list) or not isinstance(gallery_ids, list):
        raise ValueError(f"invalid FashionIQ JSON schema for {category}")
    if len(gallery_ids) != len(set(gallery_ids)):
        raise ValueError(f"FashionIQ {category} val gallery contains duplicate image IDs")
    gallery_set = set(gallery_ids)
    records: list[FashionIQRecord] = []
    for index, row in enumerate(annotations):
        if not isinstance(row, dict):
            raise ValueError(f"FashionIQ {category} annotation {index} is not an object")
        reference_id = row.get("candidate")
        target_id = row.get("target")
        captions = row.get("captions")
        if not isinstance(reference_id, str) or not isinstance(target_id, str):
            raise ValueError(
                f"FashionIQ {category} annotation {index} lacks candidate/target"
            )
        if not isinstance(captions, list) or not all(
            isinstance(caption, str) for caption in captions
        ):
            raise ValueError(f"FashionIQ {category} annotation {index} has bad captions")
        if reference_id not in gallery_set or target_id not in gallery_set:
            raise ValueError(
                f"FashionIQ {category} annotation {index} references an image "
                "outside its validation gallery"
            )
        record = FashionIQRecord(
            reference_id=reference_id,
            target_id=target_id,
            captions=tuple(captions),  # type: ignore[arg-type]
        )
        # Validate caption composition while loading, before expensive model inference.
        _ = record.text
        records.append(record)
    image_root = root / "images"
    gallery_paths = [image_root / f"{image_id}.png" for image_id in gallery_ids]
    missing = [path for path in gallery_paths if not path.is_file()]
    if missing:
        preview = "\n".join(str(path) for path in missing[:10])
        raise FileNotFoundError(
            f"FashionIQ {category} is missing {len(missing)} gallery images; "
            f"first paths:\n{preview}"
        )
    return [str(image_id) for image_id in gallery_ids], gallery_paths, records


def cache_feature_matrix(
    path: str | Path,
    progress_path: str | Path,
    *,
    count: int,
    batch_size: int,
    encode: Callable[[int, int], np.ndarray] | None = None,
    encode_batches: Callable[[int], Iterator[np.ndarray]] | None = None,
    description: str,
) -> None:
    """Build a normalized float32 NPY matrix with resumable row progress."""

    if count <= 0 or batch_size <= 0:
        raise ValueError("feature count and batch size must be positive")
    if (encode is None) == (encode_batches is None):
        raise ValueError("provide exactly one of encode or encode_batches")
    path = Path(path)
    progress_path = Path(progress_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    start = 0
    features: np.memmap | None = None
    if path.is_file() or progress_path.is_file():
        if not path.is_file() or not progress_path.is_file():
            raise RuntimeError(
                f"incomplete cache bookkeeping for {description}; use --force-cache"
            )
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        features = np.load(path, mmap_mode="r+")
        start = int(progress.get("next_index", -1))
        if (
            features.ndim != 2
            or features.shape[0] != count
            or start < 0
            or start > count
        ):
            raise ValueError(f"invalid resumable cache for {description}")
    # Start the producer only after checking progress, so resumed evaluations do
    # not decode cached rows and completed caches do not start any workers.
    stream = None
    try:
        if encode_batches is not None and start < count:
            stream = iter(encode_batches(start))
        for begin in tqdm(
            range(start, count, batch_size),
            total=(count + batch_size - 1) // batch_size,
            initial=(start + batch_size - 1) // batch_size,
            desc=description,
        ):
            end = min(begin + batch_size, count)
            if stream is not None:
                try:
                    values = next(stream)
                except StopIteration as error:
                    raise ValueError(f"incomplete feature stream for {description}") from error
            else:
                values = encode(begin, end)
            encoded = normalize_numpy(np.asarray(values, dtype=np.float32))
            if encoded.ndim != 2 or encoded.shape[0] != end - begin:
                raise ValueError(
                    f"invalid feature batch for {description}: expected {end - begin} rows, "
                    f"got {encoded.shape}"
                )
            if features is None:
                features = np.lib.format.open_memmap(
                    path,
                    mode="w+",
                    dtype=np.float32,
                    shape=(count, encoded.shape[1]),
                )
            if encoded.shape[1] != features.shape[1]:
                raise ValueError(f"embedding dimension changed while caching {description}")
            features[begin:end] = encoded
            features.flush()
            write_json_atomic(
                progress_path,
                {"next_index": end, "count": count, "complete": end == count},
            )
    finally:
        if stream is not None and hasattr(stream, "close"):
            stream.close()
    if features is None:
        raise ValueError(f"cannot build an empty cache for {description}")


def normalize_numpy(values: np.ndarray) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError(f"expected a feature matrix, got shape {values.shape}")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or bool((norms <= 1e-12).any()):
        raise ValueError("model produced non-finite or zero-norm features")
    return values / norms


@torch.inference_mode()
def evaluate_feature_files(
    *,
    gallery_path: str | Path,
    query_path: str | Path,
    gallery_ids: Sequence[str],
    records: Sequence[FashionIQRecord],
    device: torch.device,
    score_batch_size: int,
    recall_k: Iterable[int] = FASHIONIQ_RECALL_K,
) -> dict[str, float | int]:
    if score_batch_size <= 0:
        raise ValueError("score_batch_size must be positive")
    cutoffs = tuple(sorted(set(int(value) for value in recall_k)))
    if not cutoffs or min(cutoffs) <= 0:
        raise ValueError("recall cutoffs must be positive")
    gallery = np.load(Path(gallery_path), mmap_mode="r")
    queries = np.load(Path(query_path), mmap_mode="r")
    if gallery.ndim != 2 or gallery.shape[0] != len(gallery_ids):
        raise ValueError("FashionIQ gallery cache does not match gallery IDs")
    if queries.ndim != 2 or queries.shape[0] != len(records):
        raise ValueError("FashionIQ query cache does not match annotations")
    if gallery.shape[1] != queries.shape[1]:
        raise ValueError("FashionIQ gallery/query embedding dimensions differ")
    if max(cutoffs) > len(gallery_ids):
        raise ValueError("recall cutoff exceeds FashionIQ gallery size")

    image_index = {image_id: index for index, image_id in enumerate(gallery_ids)}
    targets = np.asarray(
        [image_index[record.target_id] for record in records], dtype=np.int64
    )
    hits = {cutoff: 0 for cutoff in cutoffs}
    gallery_device = torch.from_numpy(
        np.array(gallery, dtype=np.float32, copy=True)
    ).to(device)
    max_cutoff = max(cutoffs)
    for begin in tqdm(
        range(0, len(records), score_batch_size),
        desc="FashionIQ scoring",
    ):
        end = min(begin + score_batch_size, len(records))
        query_device = torch.from_numpy(
            np.array(queries[begin:end], dtype=np.float32, copy=True)
        ).to(device)
        top_indices = torch.topk(
            query_device @ gallery_device.t(),
            k=max_cutoff,
            dim=1,
        ).indices.cpu().numpy()
        batch_targets = targets[begin:end, None]
        for cutoff in cutoffs:
            hits[cutoff] += int((top_indices[:, :cutoff] == batch_targets).any(axis=1).sum())
    query_count = len(records)
    metrics: dict[str, float | int] = {
        "num_queries": query_count,
        "gallery_size": len(gallery_ids),
    }
    metrics.update(
        {
            f"recall_at_{cutoff}": round(100.0 * hits[cutoff] / query_count, 4)
            for cutoff in cutoffs
        }
    )
    return metrics


def aggregate_category_metrics(
    *,
    model_name: str,
    model_source: str,
    category_metrics: dict[str, dict[str, float | int]],
) -> dict:
    missing = set(FASHIONIQ_CATEGORIES) - set(category_metrics)
    if missing:
        raise ValueError(f"missing FashionIQ category metrics: {sorted(missing)}")
    macro = {
        f"recall_at_{cutoff}": round(
            sum(
                float(category_metrics[category][f"recall_at_{cutoff}"])
                for category in FASHIONIQ_CATEGORIES
            )
            / len(FASHIONIQ_CATEGORIES),
            4,
        )
        for cutoff in FASHIONIQ_RECALL_K
    }
    total_queries = sum(
        int(category_metrics[category]["num_queries"])
        for category in FASHIONIQ_CATEGORIES
    )
    micro = {
        f"recall_at_{cutoff}": round(
            sum(
                float(category_metrics[category][f"recall_at_{cutoff}"])
                * int(category_metrics[category]["num_queries"])
                for category in FASHIONIQ_CATEGORIES
            )
            / total_queries,
            4,
        )
        for cutoff in FASHIONIQ_RECALL_K
    }
    return {
        "schema_version": 1,
        "protocol": {
            "dataset": "FashionIQ",
            "split": "val",
            "categories": list(FASHIONIQ_CATEGORIES),
            "caption_composition": "strip('.?, ') then caption_1 + ' and ' + caption_2",
            "category_specific_gallery": True,
            "reference_image_excluded": False,
            "standard_summary": "macro average over dress, shirt, and toptee",
        },
        "model": model_name,
        "model_source": model_source,
        "num_queries": total_queries,
        "categories": category_metrics,
        "macro_average": macro,
        "micro_average": micro,
    }
