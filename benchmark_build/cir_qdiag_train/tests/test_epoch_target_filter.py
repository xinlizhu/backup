import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from test_live_rank_training import Processor
from test_loss_and_model import FakeTrainableBackbone

from cir_qdiag_train import on_policy_mining as mining
from cir_qdiag_train import on_policy_trainer as trainer
from cir_qdiag_train.model import ComposedRetrievalModel
from cir_qdiag_train.natural_rank_loss import natural_candidate_rank_loss
from cir_qdiag_train.on_policy_config import OnPolicyDistillationConfig, load_on_policy_config
from cir_qdiag_train.on_policy_data import CIRRPositiveRecord

PROJECT = Path(__file__).parents[1]
NAMES = ("kl_all_inject", "kl_top10_natural", "kl_all_natural", "cmse_top10_inject")


def test_four_configs_match_original_budget():
    baseline = load_on_policy_config(PROJECT / "configs/opd_clip_l_s2_k32_reverse_t005.yaml")
    for name in NAMES:
        c = load_on_policy_config(PROJECT / f"configs/epoch_control_{name}.yaml")
        assert c.model == baseline.model
        assert c.optimizer == baseline.optimizer
        assert c.data == baseline.data
        assert not c.mining.live_reservoir_selection
        assert c.mining.top_k == 32 and c.mining.refresh_mode == "every_epoch"
        assert c.mining.force_target_insertion == name.endswith("inject")
        assert c.objective.teacher_target_max_rank == (10 if "top10" in name else None)


def test_invalid_natural_infonce_rejected():
    c = load_on_policy_config(PROJECT / "configs/epoch_control_kl_all_natural.yaml").model_dump()
    c["objective"].update(loss_type="candidate_infonce", rank_weight=0)
    with pytest.raises(ValueError, match="no target insertion"):
        OnPolicyDistillationConfig.model_validate(c)


@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_natural_loss_retains_missing_targets_and_uses_external_filter(direction):
    s = torch.tensor([[0.1, 0.3, 0.2], [0.2, 0.4, 0.1]], requires_grad=True)
    t = torch.tensor([[0.5, 0.1, 0.2], [0.1, 0.4, 0.3]])
    # First target is absent but passes the external teacher eligibility check.
    ranks = torch.tensor([1, 12])
    loss, metrics = natural_candidate_rank_loss(
        student_scores=s,
        teacher_scores=t,
        target_positions=torch.tensor([-1, 1]),
        teacher_filter_target_ranks=ranks,
        temperature=0.05,
        direction=direction,
        max_teacher_rank=10,
    )
    sl, tl = (x[:1].log_softmax(-1) for x in (s / 0.05, t / 0.05))
    expected = ((sl.exp() * (sl - tl)) if direction == "reverse" else (tl.exp() * (tl - sl))).sum()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert s.grad[0].abs().sum() > 0
    assert s.grad[1].abs().sum() == 0
    assert metrics["valid_teacher_rate"] == 0.5
    assert metrics["teacher_target_top1_rate"] == 0.5
    zero, _ = natural_candidate_rank_loss(
        student_scores=s,
        teacher_scores=t,
        target_positions=torch.tensor([-1, 1]),
        teacher_filter_target_ranks=ranks + 32,
        temperature=0.05,
        direction=direction,
        max_teacher_rank=10,
    )
    zero.backward()
    assert zero.item() == 0


def test_raw_candidate_layout_does_not_inject_or_reorder():
    raw = np.array([[4, 3, 2], [2, 1, 4]])
    for inject in (True, False):
        out, found = np.empty_like(raw), np.zeros(2, dtype=bool)
        mining._force_targets_for_slice(
            candidate_indices=out,
            raw_target_in_topk=found,
            raw_topk=raw,
            target_indices=np.array([0, 1]),
            begin=0,
            top_k=3,
            force_targets=inject,
        )
        assert found.tolist() == [False, True]
        np.testing.assert_array_equal(out, [[0, 4, 3], [1, 2, 4]] if inject else raw)


@pytest.mark.parametrize("name", NAMES)
def test_epoch_train_cache_filter_and_resume(tmp_path, monkeypatch, name):
    torch.manual_seed(4)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    c = load_on_policy_config(PROJECT / f"configs/epoch_control_{name}.yaml")
    c.experiment.output_dir = tmp_path / "run"
    c.data.num_workers = 0
    c.mining.top_k = 3
    c.mining.image_batch_size = 4
    c.optimizer.epochs = 1
    c.optimizer.batch_size_per_gpu = 2
    c.optimizer.log_every_steps = 1
    c.optimizer.mixed_precision = "no"
    ids = [str(i) for i in range(8)]
    paths = []
    for i in range(8):
        p = tmp_path / f"{i}.png"
        Image.new("RGB", (2, 2), (i * 20 + 1, i * 10 + 3, i * 5 + 2)).save(p)
        paths.append(p)
    records = [
        CIRRPositiveRecord(
            pair_id=i,
            reference_id=str(i + 4),
            reference_path=paths[i + 4],
            target_id=str(i),
            target_path=paths[i],
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
    monkeypatch.setattr(trainer, "load_image_map", lambda *a: (ids, paths))
    monkeypatch.setattr(trainer, "load_cirr_positive_records", lambda *a: records)
    monkeypatch.setattr(
        ComposedRetrievalModel, "from_pretrained", lambda *a, **kw: (model, Processor())
    )
    gallery = torch.nn.functional.normalize(torch.randn(8, 4), dim=-1).numpy()
    queries = torch.nn.functional.normalize(torch.randn(4, 4), dim=-1).numpy()
    monkeypatch.setattr(mining, "_load_teacher_cache", lambda **kw: (gallery, queries))
    before = model.backbone.image_projection.weight.detach().clone()
    trainer.train_on_policy_distillation(c)
    assert not torch.equal(before, model.backbone.image_projection.weight)
    directory = c.experiment.output_dir / "on_policy/epoch_001"
    candidates = np.load(directory / "candidate_indices.npy")
    scores = np.load(directory / "teacher_scores.npy")
    expected = np.einsum("bd,bkd->bk", queries, gallery[candidates]).astype(np.float16)
    np.testing.assert_allclose(scores, expected, atol=5e-4)
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["target_forced_into_candidates"] == c.mining.force_target_insertion
    if not c.mining.force_target_insertion:
        reference = np.stack(
            [mining.force_target_into_topk(row, i, 3)[0] for i, row in enumerate(candidates)]
        )
        expected = np.einsum("bd,bkd->bk", queries, gallery[reference]).astype(np.float16)
        ranks = 1 + (expected[:, 1:] > expected[:, :1]).sum(1)
        np.testing.assert_array_equal(np.load(directory / "teacher_filter_target_ranks.npy"), ranks)
        assert metadata["candidate_layout"] == "natural_student_top_k"
    # Exercise reuse of the correct mode and completed-epoch optimizer/scheduler resume.
    _, cached = mining.refresh_on_policy_candidates(
        config=c,
        model=model,
        processor=Processor(),
        records=records,
        image_ids=ids,
        image_paths=paths,
        epoch=1,
        global_step=0,
        device=torch.device("cpu"),
    )
    assert cached == metadata
    c.optimizer.epochs = 2
    trainer.train_on_policy_distillation(c, resume_from=c.experiment.output_dir / "last.pt")
    assert (c.experiment.output_dir / "checkpoint_epoch_002.pt").is_file()


def test_new_runner_resume_and_no_baseline_training(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "epoch_runner_test", PROJECT / "scripts/run_epoch_target_filter.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    c = load_on_policy_config(module.runner.CONFIG_PATHS[NAMES[1]])
    c.experiment.output_dir = tmp_path / NAMES[1]
    c.experiment.output_dir.mkdir()
    (c.experiment.output_dir / "last.pt").touch()
    monkeypatch.setattr(module.runner, "load_on_policy_config", lambda path: c)
    calls = []

    def execute(argv, log, cwd=None):
        args = list(map(str, argv))
        calls.append(args)

        def arg(key):
            return Path(args[args.index(key) + 1])

        if args[1].endswith("train_on_policy_distillation.py"):
            assert arg("--resume-from").name == "last.pt"
            for e in range(1, 5):
                (c.experiment.output_dir / f"checkpoint_epoch_{e:03d}.pt").touch()
        elif "--output" in args:
            arg("--output").write_text("{}")
        else:
            p = arg("--output-dir")
            p.mkdir(parents=True)
            (p / "metrics.json").write_text("{}")

    monkeypatch.setattr(module.runner, "execute", execute)
    module.runner.run_one(NAMES[1], extended=False)
    module.runner.run_one(NAMES[1], extended=False)
    assert len(calls) == 9  # One training run, four CIRR and four FashionIQ.
    monkeypatch.setattr(module, "ROOT", tmp_path)
    module.summarize()
    report = (tmp_path / "RESULTS.md").read_text()
    assert "existing_kl_top10_inject" in report
    assert all(name in report for name in NAMES)
