import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cir_qdiag_train.qwen_on_policy_trainer import qwen_candidate_loss, refresh_qwen_candidates

PROJECT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "main9_test", PROJECT / "scripts/run_main_comparison_e4.py"
)
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


def test_nine_configs():
    outputs = set()
    for name in suite.RUNS:
        config = suite.load(name)
        assert config.data.num_workers == 8
        outputs.add(config.experiment.output_dir)
        assert config.optimizer.epochs == 4
        assert config.mining.top_k == 32
        assert config.mining.force_target_insertion
        assert not config.mining.live_reservoir_selection
        if "oprd" in name:
            assert config.mining.candidate_source == "student"
            assert config.objective.rank_kl_direction == "reverse"
            assert config.objective.teacher_target_max_rank == 10
        else:
            assert config.mining.candidate_source == "random"
            assert not config.objective.uses_teacher_filter
            assert config.objective.needs_teacher_scores == ("cmse" in name)
    assert len(outputs) == 9


@pytest.mark.parametrize("name", ["qwen2_infonce", "qwen2_cmse_s2", "qwen2_oprd_s2"])
def test_loss_math(name):
    config = suite.load(name)
    s = torch.tensor([[0.1, 0.2, -0.1]], requires_grad=True)
    t = torch.tensor([[0.3, 0.1, 0.2]])
    loss, _ = qwen_candidate_loss(
        config=config,
        student_scores=s,
        teacher_scores=t,
        raw_target_in_topk=torch.tensor([False]),
        teacher_target_top1=torch.tensor([True]),
        rank_weight=config.objective.rank_weight,
        centered_response_weight=0.0,
        rank_temperature=config.objective.rank_temperature,
        centered_response_beta=0.01,
        require_teacher_target_top1=False,
        teacher_target_max_rank=config.objective.teacher_target_max_rank,
        rank_kl_direction="reverse",
    )
    if "infonce" in name:
        expected = torch.nn.functional.cross_entropy(s / 0.07, torch.tensor([0]))
    elif "cmse" in name:
        residual = s - t
        expected = ((residual - residual.mean(-1, keepdim=True)) ** 2).mean()
    else:
        log_s, log_t = (s / 0.05).log_softmax(-1), (t / 0.05).log_softmax(-1)
        expected = (log_s.exp() * (log_s - log_t)).sum(-1).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(s.grad).all() and s.grad.abs().sum() > 0


@pytest.mark.parametrize("name", ["qwen2_infonce", "qwen2_cmse_s2"])
def test_random_mining_never_encodes_student(name, tmp_path, monkeypatch):
    import cir_qdiag_train.qwen_on_policy_trainer as trainer

    config = suite.load(name)
    config.experiment.output_dir = tmp_path
    records = [SimpleNamespace(reference_id="0", target_id="1") for _ in range(5)]
    calls = []

    def teacher(**kwargs):
        calls.append(kwargs)
        return np.ones((5, 32), dtype=np.float32), np.ones(5, dtype=bool)

    monkeypatch.setattr(trainer, "_score_teacher_candidates", teacher)
    # No query encoder exists: any accidental Student retrieval fails the test.
    student = SimpleNamespace(eval=lambda: None)
    kwargs = dict(
        config=config,
        student=student,
        records=records,
        image_ids=[str(i) for i in range(50)],
        gallery=np.eye(50),
        epoch=1,
        global_step=0,
        device=torch.device("cpu"),
    )
    path, metadata = refresh_qwen_candidates(**kwargs)
    candidates = np.load(path / "candidate_indices.npy")
    assert (candidates[:, 0] == 1).all()
    assert not (candidates == 0).any()
    assert all(len(set(row)) == 32 for row in candidates)
    assert len(calls) == int("cmse" in name)
    assert metadata["candidate_source"] == "random"
    refresh_qwen_candidates(**kwargs)
    assert len(calls) == int("cmse" in name)  # cached second call
    path2, _ = refresh_qwen_candidates(**{**kwargs, "epoch": 2})
    assert not np.array_equal(candidates, np.load(path2 / "candidate_indices.npy"))


def test_qwen_runner_resume_and_skip(tmp_path, monkeypatch):
    config = suite.load("qwen2_infonce")
    config.experiment.output_dir = tmp_path
    (tmp_path / "last.pt").touch()
    monkeypatch.setattr(suite, "load", lambda name: config)
    commands = []

    def execute(argv, log, **kwargs):
        commands.append([str(x) for x in argv])
        (tmp_path / "checkpoint_epoch_004.pt").touch()

    monkeypatch.setattr(suite.runner, "execute", execute)
    suite.run_qwen("qwen2_infonce")
    assert len(commands) == 5
    assert "--resume-from" in commands[0]
    assert "--adapter" in commands[-1]
    for path in [
        "cirr_val/val_metrics.json",
        "fashioniq/metrics.json",
        "pinpoint_official/MAIN9-qwen2_infonce/official_metrics.csv",
        "qdiag/MAIN9-qwen2_infonce/summary.csv",
    ]:
        p = tmp_path / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    suite.run_qwen("qwen2_infonce")
    assert len(commands) == 5


def test_qwen_real_teacher_scoring(tmp_path, monkeypatch):
    import cir_qdiag_train.on_policy_mining as mining

    config = suite.load("qwen2_cmse_s2")
    config.experiment.output_dir = tmp_path
    config.optimizer.mixed_precision = "no"
    gallery = np.eye(40, dtype=np.float32)
    monkeypatch.setattr(mining, "_load_teacher_cache", lambda **kwargs: (gallery, gallery[[1, 2]]))
    path, _ = refresh_qwen_candidates(
        config=config,
        student=SimpleNamespace(eval=lambda: None),
        records=[SimpleNamespace(reference_id="0", target_id=str(i)) for i in [1, 2]],
        image_ids=[str(i) for i in range(40)],
        gallery=gallery,
        epoch=1,
        global_step=0,
        device=torch.device("cpu"),
    )
    scores = np.load(path / "teacher_scores.npy")
    assert (scores[:, 0] == 1).all()
    assert (scores[:, 1:] == 0).all()


@pytest.mark.parametrize("name", ["qwen2_infonce", "qwen2_cmse_s2", "qwen2_oprd_s2"])
@pytest.mark.parametrize("instruct_init", [False, True])
def test_qwen_training_cpu_smoke(name, tmp_path, monkeypatch, instruct_init):
    import cir_qdiag_train.qwen_on_policy_trainer as trainer

    config = suite.load(name)
    if instruct_init:
        from cir_qdiag_train.qwen_on_policy_config import load_qwen_on_policy_config
        config = load_qwen_on_policy_config(PROJECT / f"configs/qwen2_instruct_query_{name.removeprefix('qwen2_')}.yaml")
    config.data.num_workers = 0  # fake Student has no tokenizer; workers tested separately
    config.experiment.output_dir = tmp_path
    config.optimizer.epochs = 1
    config.optimizer.batch_size_per_gpu = 2
    config.optimizer.gradient_accumulation_steps = 1
    config.optimizer.mixed_precision = "no"
    config.optimizer.log_every_steps = 1
    records = [
        SimpleNamespace(reference_id="0", target_id=str(i), reference_path="unused", text="edit")
        for i in [1, 2, 3, 4]
    ]
    gallery = np.random.default_rng(42).normal(size=(40, 4)).astype(np.float32)
    gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)

    class Student(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([0.2, 0.3, 0.4, 0.5]))

        def trainable_parameters(self):
            return [self.weight]

        def parameter_summary(self):
            return {}

        def prepare_query_batch(self, paths, texts):
            return len(paths)

        def encode_prepared_queries(self, count, requires_grad):
            return torch.nn.functional.normalize(self.weight, dim=0).expand(count, -1)

    saved = []

    def save(path, **kwargs):
        saved.append(kwargs["student"].weight.detach().clone())
        path.touch()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(trainer, "QwenQueryLoRAStudent", Student)
    monkeypatch.setattr(trainer, "load_image_map", lambda *args: ([str(i) for i in range(40)], []))
    monkeypatch.setattr(trainer, "load_cirr_positive_records", lambda *args: records)
    monkeypatch.setattr(trainer, "_load_student_gallery", lambda *args: gallery)
    monkeypatch.setattr(trainer, "save_qwen_checkpoint", save)

    def teacher(**kwargs):
        scores = np.zeros((4, 32), dtype=np.float32)
        scores[:, 0] = 0.8
        return scores, np.ones(4, dtype=bool)

    monkeypatch.setattr(trainer, "_score_teacher_candidates", teacher)
    result = trainer.train_qwen_on_policy_distillation(config)
    assert result.is_file()
    assert len(saved) == 2
    assert not torch.equal(saved[0], torch.tensor([0.2, 0.3, 0.4, 0.5]))
