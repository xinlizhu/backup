"""CPU regressions for ordered multiprocessing preprocessing and resumed eval."""

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from cir_qdiag_train.config import EvalConfig
from cir_qdiag_train.evaluation import evaluate_cirr
from cir_qdiag_train.fashioniq_evaluation import cache_feature_matrix
from cir_qdiag_train.inference_data import build_inference_loader, inference_loader_settings
from cir_qdiag_train.on_policy_config import load_on_policy_config
from cir_qdiag_train.on_policy_mining import _encode_student_gallery


class Processor:
    def __init__(self, report_pid=False):
        self.report_pid = report_pid

    def __call__(self, *, images=None, text=None, **kwargs):
        if images is not None:
            values = torch.from_numpy(np.stack([np.array(image) for image in images])).float()
            values = values.permute(0, 3, 1, 2).contiguous()
            if self.report_pid:
                values.fill_(os.getpid())
            return {"pixel_values": values}
        lengths = [len(row.split()) for row in text]
        ids = torch.zeros(len(text), max(lengths), dtype=torch.long)
        for i, row in enumerate(text):
            ids[i, :lengths[i]] = torch.tensor([int(token) for token in row.split()])
        return {"input_ids": ids, "attention_mask": (ids != 0).long()}


class Model(torch.nn.Module):
    def encode_image(self, pixels):
        return torch.nn.functional.normalize(pixels.mean(dim=(2, 3)), dim=-1)

    def encode_reference(self, pixels):
        return self.encode_image(pixels)

    def encode_text(self, ids, mask):
        return torch.nn.functional.one_hot(ids[:, 0], num_classes=3).float()

    def compose(self, reference, text):
        return torch.nn.functional.normalize(reference + 10 * text, dim=-1)


def make_images(tmp_path, count=5):
    paths = []
    for index in range(count):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (4, 4), (index + 1, index + 3, index + 5)).save(path)
        paths.append(path)
    return paths


@pytest.mark.parametrize("workers", [0, 2])
def test_loader_preserves_pixels_tokens_order_tail_and_rng(tmp_path, workers):
    paths = make_images(tmp_path)
    texts = ["1", "2 1", "1 2 1", "2", "1 1"]
    processor = Processor()
    state = torch.get_rng_state().clone()
    loader = build_inference_loader(
        image_paths=paths, texts=texts, processor=processor,
        batch_size=3, num_workers=workers, prefetch_factor=1, start=1,
    )
    batches = list(loader)
    assert torch.equal(state, torch.get_rng_state())
    assert [batch["indices"].tolist() for batch in batches] == [[1, 2, 3], [4]]
    for batch in batches:
        indices = batch["indices"].tolist()
        images = [Image.open(paths[index]).convert("RGB") for index in indices]
        expected = processor(images=images)["pixel_values"]
        tokens = processor(text=[texts[index] for index in indices])
        assert torch.equal(batch["pixel_values"], expected)
        assert torch.equal(batch["input_ids"], tokens["input_ids"])
        assert torch.equal(batch["attention_mask"], tokens["attention_mask"])
        for image in images:
            image.close()


def test_processor_runs_in_multiple_child_processes(tmp_path):
    loader = build_inference_loader(
        image_paths=make_images(tmp_path, count=4), processor=Processor(report_pid=True),
        batch_size=2, num_workers=2, prefetch_factor=1,
    )
    worker_pids = {int(batch["pixel_values"][0, 0, 0, 0]) for batch in loader}
    assert len(worker_pids) == 2
    assert os.getpid() not in worker_pids


def test_local_clip_processor_is_spawn_safe_and_identical(tmp_path):
    from transformers import AutoProcessor

    model_path = Path(__file__).parents[1] / "models/clip-vit-large-patch14"
    processor = AutoProcessor.from_pretrained(
        model_path, local_files_only=True, use_fast=False,
    )
    paths = make_images(tmp_path, count=3)
    results = []
    for workers in (0, 2):
        results.append(list(build_inference_loader(
            image_paths=paths, texts=["a red shirt", "a longer blue dress", "green"],
            processor=processor, batch_size=2, num_workers=workers, prefetch_factor=1,
        )))
    for serial, parallel in zip(*results, strict=True):
        assert serial.keys() == parallel.keys()
        for key in serial:
            assert torch.equal(serial[key], parallel[key]), key


def test_inference_settings_and_config(monkeypatch):
    monkeypatch.setenv("CIR_INFERENCE_NUM_WORKERS", "4")
    monkeypatch.setenv("CIR_INFERENCE_PREFETCH_FACTOR", "1")
    assert inference_loader_settings(default_workers=8) == (4, 1)
    assert inference_loader_settings(num_workers=0, prefetch_factor=2) == (0, 2)
    assert EvalConfig().num_workers == 8
    with pytest.raises(ValueError):
        inference_loader_settings(num_workers=-1)
    with pytest.raises(ValueError):
        EvalConfig(prefetch_factor=0)


def test_bad_paths_fail_instead_of_skipping_images(tmp_path):
    loader = build_inference_loader(
        image_paths=[tmp_path / "missing.png"], processor=Processor(),
        batch_size=1, num_workers=0,
    )
    with pytest.raises(FileNotFoundError):
        list(loader)


def test_mining_gallery_matches_original_encoding(tmp_path, monkeypatch):
    monkeypatch.setenv("CIR_INFERENCE_NUM_WORKERS", "2")
    paths = make_images(tmp_path)
    processor, model = Processor(), Model()
    actual = _encode_student_gallery(
        model=model, processor=processor, image_paths=paths, batch_size=3,
        device=torch.device("cpu"), mixed_precision="no",
    )
    images = [Image.open(path).convert("RGB") for path in paths]
    expected = model.encode_image(processor(images=images)["pixel_values"])
    expected = torch.nn.functional.normalize(expected, dim=-1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for image in images:
        image.close()


def test_cirr_parallel_evaluation_matches_serial(tmp_path, monkeypatch):
    from cir_qdiag_train import evaluation

    project = Path(__file__).parents[1]
    config = load_on_policy_config(project / "configs/clip_l_reverse_rank_static_r128_e4.yaml")
    image_map = {}
    for index, color in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255))):
        Image.new("RGB", (4, 4), color).save(tmp_path / f"{index}.png")
        image_map[str(index)] = f"{index}.png"
    rows = [
        {"reference": str(i), "target_hard": str((i + 1) % 3),
         "caption": str((i + 1) % 3), "img_set": {"members": list(image_map)}}
        for i in range(3)
    ]
    config.data.image_root = tmp_path
    config.data.val_annotation_path = tmp_path / "annotations.json"
    config.data.val_image_split_path = tmp_path / "split.json"
    config.data.val_annotation_path.write_text(json.dumps(rows))
    config.data.val_image_split_path.write_text(json.dumps(image_map))
    config.optimizer.mixed_precision = "no"
    config.evaluation.batch_size = 2
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        evaluation.ComposedRetrievalModel, "from_pretrained",
        lambda *args, **kwargs: (Model(), Processor()),
    )
    results = []
    for workers in (0, 2):
        monkeypatch.setenv("CIR_INFERENCE_NUM_WORKERS", str(workers))
        results.append(evaluate_cirr(
            config, checkpoint_path=None, output_path=tmp_path / f"metrics_{workers}.json",
        ))
    assert results[0] == results[1]
    assert results[0]["recall_at_1"] == results[0]["group_recall_at_1"] == 100.0


def test_streamed_cache_resumes_and_closes_producer(tmp_path):
    path, progress = tmp_path / "features.npy", tmp_path / "progress.json"
    values = np.eye(5, dtype=np.float32)
    starts, closed = [], []

    def interrupted(start):
        starts.append(start)
        try:
            yield values[:2]
            raise RuntimeError("test interruption")
        finally:
            closed.append(True)

    with pytest.raises(RuntimeError, match="test interruption"):
        cache_feature_matrix(
            path, progress, count=5, batch_size=2,
            encode_batches=interrupted, description="interrupted",
        )
    assert json.loads(progress.read_text())["next_index"] == 2
    assert closed == [True]

    def resume(start):
        starts.append(start)
        try:
            for i in range(start, 5, 2):
                yield values[i:i + 2]
        finally:
            closed.append(True)

    for _ in range(2):
        cache_feature_matrix(
            path, progress, count=5, batch_size=2, encode_batches=resume, description="resume",
        )
    np.testing.assert_array_equal(np.load(path), values)
    assert starts == [0, 2]  # Completed cache never creates a producer.
    assert closed == [True, True]


def test_fashioniq_parallel_features_match_original_forward(tmp_path):
    path = Path(__file__).parents[1] / "scripts/evaluate_fashioniq_clip.py"
    spec = importlib.util.spec_from_file_location("fashion_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paths = make_images(tmp_path)
    texts = ["1", "2 1", "1 2 1", "2", "1 1"]
    model, processor = Model(), Processor()
    for query_texts in (None, texts):
        batches = list(module._iter_encoded_batches(
            model=model, processor=processor, image_paths=paths, texts=query_texts,
            batch_size=3, num_workers=2, prefetch_factor=1, pin_memory=False,
            start=1, device=torch.device("cpu"), mixed_precision="no",
        ))
        images = [Image.open(path).convert("RGB") for path in paths[1:]]
        features = model.encode_image(processor(images=images)["pixel_values"])
        if query_texts is not None:
            tokens = processor(text=texts[1:])
            features = model.compose(
                features, model.encode_text(tokens["input_ids"], tokens["attention_mask"]),
            )
        np.testing.assert_array_equal(np.concatenate(batches), features.numpy())
        for image in images:
            image.close()
