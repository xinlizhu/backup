"""TG-CIR evaluation with bounded multi-worker decoding; no target-guided teacher."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from cir_qdiag_train.fashioniq_evaluation import (
    FASHIONIQ_CATEGORIES,
    aggregate_category_metrics,
    cache_feature_matrix,
    evaluate_feature_files,
    load_fashioniq_val,
)
from cir_qdiag_train.inference_data import build_inference_loader
from cir_qdiag_train.on_policy_data import load_cirr_positive_records, load_image_map
from cir_qdiag_train.on_policy_mining import write_json_atomic
from cir_qdiag_train.tg_cir_config import model_label


def cirr_split(config, split):
    data = config["data"]
    ids, paths = load_image_map(
        Path(data[f"{split}_image_split_path"]),
        Path(data["image_root"]),
    )
    records = load_cirr_positive_records(Path(data[f"{split}_annotation_path"]), ids, paths)
    return ids, paths, records


@torch.inference_mode()
def feature_batches(model, processor, paths, texts, config, device, start=0):
    model.eval()
    options = config["evaluation"]
    loader = build_inference_loader(
        image_paths=paths,
        texts=texts,
        processor=processor,
        start=start,
        batch_size=options["batch_size"],
        num_workers=options["num_workers"],
        prefetch_factor=options["prefetch_factor"],
        pin_memory=device.type == "cuda",
    )
    for batch in loader:
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            if texts is None:
                embedding = model.extract_retrieval_target(pixels)
            else:
                embedding = model.extract_retrieval_compose(
                    pixels,
                    batch["input_ids"].to(device, non_blocking=True),
                )
        yield embedding.float().cpu().numpy()


def cache_embeddings(model, processor, paths, texts, config, device, directory, name):
    directory = Path(directory)
    cache_feature_matrix(
        directory / f"{name}_features.npy",
        directory / f"{name}_progress.json",
        count=len(paths),
        batch_size=config["evaluation"]["batch_size"],
        encode_batches=lambda start: feature_batches(
            model,
            processor,
            paths,
            texts,
            config,
            device,
            start,
        ),
        description=f"TG-CIR {directory.name} {name}",
    )
    return directory / f"{name}_features.npy"


def score_cirr(gallery, queries, image_ids, records, annotations, device):
    image_index = {image_id: index for index, image_id in enumerate(image_ids)}
    ks, group_ks = (1, 5, 10, 50), (1, 2, 3)
    hits, group_hits = dict.fromkeys(ks, 0), dict.fromkeys(group_ks, 0)
    gallery = torch.as_tensor(np.array(gallery, dtype=np.float32), device=device)
    for begin in tqdm(range(0, len(records), 512), desc="TG-CIR CIRR scores"):
        rows = records[begin : begin + 512]
        query = torch.as_tensor(np.array(queries[begin : begin + 512]), device=device)
        scores = query @ gallery.T
        refs = torch.tensor([image_index[row.reference_id] for row in rows], device=device)
        scores.scatter_(1, refs[:, None], -torch.inf)
        top = scores.topk(min(50, len(image_ids)), dim=-1).indices.cpu().numpy()
        for offset, row in enumerate(rows):
            target = image_index[row.target_id]
            for k in ks:
                hits[k] += int(target in top[offset, :k])
            members = annotations[begin + offset]["img_set"]["members"]
            group = [image_index[x] for x in members if x != row.reference_id]
            if target not in group:
                raise ValueError("CIRR validation group lacks target")
            order = scores[offset, group].argsort(descending=True).cpu().tolist()
            ranked = [group[index] for index in order]
            for k in group_ks:
                group_hits[k] += int(target in ranked[:k])
    return {
        "num_queries": len(records),
        **{f"recall_at_{k}": round(100 * hits[k] / len(records), 4) for k in ks},
        **{f"group_recall_at_{k}": round(100 * group_hits[k] / len(records), 4) for k in group_ks},
    }


def evaluate_cirr(model, processor, config, device, directory):
    ids, paths, records = cirr_split(config, "val")
    gallery = cache_embeddings(model, processor, paths, None, config, device, directory, "gallery")
    queries = cache_embeddings(
        model,
        processor,
        [row.reference_path for row in records],
        [row.text for row in records],
        config,
        device,
        directory,
        "query",
    )
    annotations = json.loads(Path(config["data"]["val_annotation_path"]).read_text())
    metrics = score_cirr(
        np.load(gallery, mmap_mode="r"),
        np.load(queries, mmap_mode="r"),
        ids,
        records,
        annotations,
        device,
    )
    write_json_atomic(Path(directory) / "metrics.json", metrics)
    return metrics


def evaluate_fashioniq(model, processor, config, device, directory, checkpoint):
    results = {}
    for category in FASHIONIQ_CATEGORIES:
        ids, paths, records = load_fashioniq_val(config["evaluation"]["fashioniq_root"], category)
        lookup = dict(zip(ids, paths, strict=True))
        folder = Path(directory) / category
        gallery = cache_embeddings(model, processor, paths, None, config, device, folder, "gallery")
        queries = cache_embeddings(
            model,
            processor,
            [lookup[row.reference_id] for row in records],
            [row.text for row in records],
            config,
            device,
            folder,
            "query",
        )
        results[category] = evaluate_feature_files(
            gallery_path=gallery,
            query_path=queries,
            gallery_ids=ids,
            records=records,
            device=device,
            score_batch_size=1024,
        )
        write_json_atomic(folder / "metrics.json", results[category])
    metrics = aggregate_category_metrics(
        model_name=model_label(config),
        model_source=str(checkpoint),
        category_metrics=results,
    )
    write_json_atomic(Path(directory) / "metrics.json", metrics)
    return metrics
