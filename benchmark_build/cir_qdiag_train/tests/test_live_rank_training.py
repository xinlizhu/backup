"""CPU integration: epoch mining -> live selection -> aligned Teacher KD -> resume."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from test_loss_and_model import FakeImageProcessor, FakeTrainableBackbone

from cir_qdiag_train import on_policy_mining as mining
from cir_qdiag_train import on_policy_trainer as trainer
from cir_qdiag_train.model import ComposedRetrievalModel
from cir_qdiag_train.on_policy_config import load_on_policy_config
from cir_qdiag_train.on_policy_data import CIRRPositiveRecord


class Processor(FakeImageProcessor):
    def __call__(self, *, text=None, **kwargs):
        if text is None:
            return super().__call__(**kwargs)
        return {
            "input_ids": torch.tensor([[int(t), int(t) + 1] for t in text]),
            "attention_mask": torch.ones(len(text), 2, dtype=torch.long),
        }


@pytest.mark.parametrize("filtered", [False, True])
def test_live_rank_training_and_epoch_resume(tmp_path, monkeypatch, filtered):
    torch.manual_seed(4)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    project = Path(__file__).parents[1]
    config = load_on_policy_config(project / "configs/clip_l_reverse_rank_opd_r128_e4.yaml")
    config.experiment.output_dir = tmp_path / "run"
    config.objective.teacher_target_max_rank = 2 if filtered else None
    config.mining.reservoir_top_k = 5
    config.mining.top_k = 3
    config.mining.pixel_cache_dtype = "float32"
    config.mining.pixel_cache_workers = 0
    config.data.num_workers = 0
    config.optimizer.epochs = 2
    config.optimizer.batch_size_per_gpu = 2
    config.optimizer.mixed_precision = "no"
    config.optimizer.log_every_steps = 1
    image_ids = [str(i) for i in range(8)]
    image_paths = []
    for i in range(8):
        path = tmp_path / f"image_{i}.png"
        Image.new("RGB", (2, 2), color=(20 * i + 1, 10 * i + 3, 5 * i + 2)).save(path)
        image_paths.append(path)
    records = [
        CIRRPositiveRecord(
            pair_id=i,
            reference_id=str(i + 4),
            reference_path=image_paths[i + 4],
            target_id=str(i),
            target_path=image_paths[i],
            text=str(i + 1),
        )
        for i in range(4)
    ]
    model = ComposedRetrievalModel(
        FakeTrainableBackbone(),
        embedding_dim=4,
        hidden_dim=8,
        dropout=0.1,
        freeze_backbone=False,
        temperature=0.07,
        learnable_temperature=False,
    )
    processor = Processor()
    monkeypatch.setattr(trainer, "load_image_map", lambda *a: (image_ids, image_paths))
    monkeypatch.setattr(trainer, "load_cirr_positive_records", lambda *a: records)
    monkeypatch.setattr(
        ComposedRetrievalModel, "from_pretrained", lambda *a, **kw: (model, processor)
    )
    gallery = torch.nn.functional.normalize(torch.randn(8, 4), dim=-1).numpy()
    queries = torch.nn.functional.normalize(torch.randn(4, 4), dim=-1).numpy()
    monkeypatch.setattr(mining, "_load_teacher_cache", lambda **kw: (gallery, queries))
    selection_calls = []
    original_select = trainer._select_live_topk_from_reservoir
    original_loss = trainer.on_policy_centered_distillation_loss
    seen_scores = []

    def select(**kwargs):
        assert not kwargs["model"].training
        assert "reservoir_teacher_scores" in kwargs["batch"]
        result = original_select(**kwargs)
        positions = result["selected_candidate_indices"].numpy()
        pair_ids = kwargs["batch"]["pair_ids"]
        expected = np.einsum("bd,bkd->bk", queries[pair_ids], gallery[positions])
        np.testing.assert_allclose(result["selected_teacher_scores"].numpy(), expected, atol=1e-6)
        seen_scores.append(result["selected_teacher_scores"].clone())
        selection_calls.append(model.backbone.image_projection.weight.detach().clone())
        return result

    def loss(**kwargs):
        assert kwargs["rank_kl_direction"] == "reverse"
        assert kwargs["teacher_target_max_rank"] == (2 if filtered else None)
        assert torch.equal(kwargs["teacher_scores"], seen_scores[-1])
        assert model.training
        return original_loss(**kwargs)

    monkeypatch.setattr(trainer, "_select_live_topk_from_reservoir", select)
    monkeypatch.setattr(trainer, "on_policy_centered_distillation_loss", loss)
    initial = model.backbone.image_projection.weight.detach().clone()
    trainer.train_on_policy_distillation(config)
    assert len(selection_calls) == 4
    assert not torch.equal(initial, model.backbone.image_projection.weight)
    assert not torch.equal(selection_calls[0], selection_calls[1])
    run = config.experiment.output_dir
    assert (run / "checkpoint_epoch_002.pt").is_file()
    assert not (run / "on_policy").exists()  # Reservoirs/Teacher subset scores remain in RAM.
    logs = [json.loads(line) for line in (run / "train_metrics.jsonl").read_text().splitlines()]
    assert all(row["rank_kl_is_reverse"] == 1 for row in logs)
    assert all("live_reservoir_raw_target_recall_at_k" in row for row in logs)
    if not filtered:
        assert all(row["valid_teacher_rate"] == 1 for row in logs)
    # A completed-epoch resume must rebuild the next reservoir with the resumed Student.
    config.optimizer.epochs = 3
    trainer.train_on_policy_distillation(config, resume_from=run / "last.pt")
    assert len(selection_calls) == 6
    assert (run / "checkpoint_epoch_003.pt").is_file()
