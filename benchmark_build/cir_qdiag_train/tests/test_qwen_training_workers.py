import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from cir_qdiag_train.qwen_on_policy_config import load_qwen_on_policy_config
from cir_qdiag_train.qwen_training_data import QueryCollator, query_batches
from cir_qdiag_train.resume_config import without_loader_settings

PROJECT = Path(__file__).resolve().parent.parent


def test_resume_loader_change_only():
    old = {
        "data": {"num_workers": 8, "prefetch_factor": 2, "image_root": "images"},
        "optimizer": {"batch_size": 32},
    }
    new = {
        "data": {"num_workers": 16, "prefetch_factor": 1, "image_root": "images"},
        "optimizer": {"batch_size": 32},
    }
    assert without_loader_settings(old) == without_loader_settings(new)
    assert old["data"]["num_workers"] == 8
    new["optimizer"]["batch_size"] = 16
    assert without_loader_settings(old) != without_loader_settings(new)
    assert without_loader_settings({"data": {}}) == without_loader_settings(
        {"data": {"num_workers": 0, "prefetch_factor": 2}}
    )


def test_qwen_zero_workers_fallback_order():
    student = SimpleNamespace(prepare_query_batch=lambda paths, texts: {"texts": texts})
    records = [SimpleNamespace(reference_path="unused", text=str(i)) for i in range(5)]
    batches = list(
        query_batches(
            student=student,
            records=records,
            order=[4, 1, 3, 0, 2],
            batch_size=2,
            workers=0,
            prefetch_factor=2,
            seed=42,
        )
    )
    assert [idx for ids, _ in batches for idx in ids] == [4, 1, 3, 0, 2]
    assert batches[-1][1]["texts"] == ["2"]


class ReportingProcessor:
    """Exercise the real vision preprocessing without loading a model/tokenizer."""

    def apply_chat_template(self, conversations, **kwargs):
        return ["query"] * len(conversations)

    def __call__(self, text, images, **kwargs):
        import numpy as np

        return {
            "pixels": torch.tensor([np.asarray(im).mean() for im in images]),
            "worker_pid": torch.tensor(os.getpid()),
        }


def test_spawn_preprocessing_order_pixels_and_rng(tmp_path):
    config = load_qwen_on_policy_config(PROJECT / "configs/main9_qwen2_infonce.yaml")
    if not Path(config.model.pretrained_model_name_or_path).is_dir():
        pytest.skip("local Qwen preprocessing code unavailable")
    records = []
    for i in range(3):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (56, 56), (i * 40, i * 40, i * 40)).save(path)
        records.append(SimpleNamespace(reference_path=path, text="edit"))
    student = SimpleNamespace(
        embedder=SimpleNamespace(processor=ReportingProcessor()),
        config=config.model,
        device=torch.device("cpu"),
    )
    collator = QueryCollator(student.embedder.processor, config.model)
    expected = collator([(i, records[i].reference_path, records[i].text) for i in [2, 0]])[1]
    state = torch.get_rng_state().clone()
    batches = list(
        query_batches(
            student=student,
            records=records,
            order=[2, 0, 1],
            batch_size=2,
            workers=2,
            prefetch_factor=1,
            seed=42,
        )
    )
    assert [i for ids, _ in batches for i in ids] == [2, 0, 1]
    torch.testing.assert_close(batches[0][1]["pixels"], expected["pixels"])
    assert all(batch["worker_pid"].item() != os.getpid() for _, batch in batches)
    assert all(batch["pixels"].device.type == "cpu" for _, batch in batches)
    assert torch.equal(state, torch.get_rng_state())
