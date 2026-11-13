import importlib.util
import json
from pathlib import Path

import pytest
import torch

from cir_qdiag_train.on_policy_config import load_on_policy_config
from cir_qdiag_train.qwen_on_policy_config import load_qwen_on_policy_config
from cir_qdiag_train.qwen_live_training import compute_live_qwen_scores

PROJECT = Path(__file__).parents[1]


@pytest.mark.parametrize("checkpoint_exists", [False, True])
def test_selected_epoch_never_trains_or_uses_later_checkpoint(tmp_path, monkeypatch, checkpoint_exists):
    spec = importlib.util.spec_from_file_location("selected_epoch_suite", PROJECT / "scripts/run_main_comparison_e4.py")
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)
    c = load_qwen_on_policy_config(PROJECT / "configs/qwen2_instruct_shared_oprd_s2.yaml")
    c.experiment.output_dir = tmp_path
    name = c.experiment.name
    monkeypatch.setattr(suite, "load", lambda _: c)
    monkeypatch.setattr(suite, "RUNS", (name,))
    monkeypatch.setattr(suite, "ROOT", tmp_path)
    monkeypatch.setitem(suite.EVALUATION_EPOCHS, name, 2)
    monkeypatch.setitem(suite.runner.CONFIG_PATHS, name, PROJECT / f"configs/{name}.yaml")
    (tmp_path / "effective_config.json").write_text(json.dumps(c.model_dump(mode="json")))
    for f in ("last.pt", "checkpoint_epoch_003.pt", "checkpoint_epoch_004.pt"):
        (tmp_path / f).touch()
    (tmp_path / "cirr_val").mkdir()
    (tmp_path / "cirr_val/val_metrics.json").write_text('{"recall_at_1": 99}')
    calls = []
    monkeypatch.setattr(suite.runner, "execute", lambda cmd, *a, **kw: calls.append(list(map(str, cmd))))
    if not checkpoint_exists:
        with pytest.raises(FileNotFoundError, match="Evaluation-only checkpoint missing"):
            suite.run_qwen(name)
        assert not calls
        return
    (tmp_path / "checkpoint_epoch_002.pt").touch()
    suite.run_qwen(name)
    assert len(calls) == 4
    for cmd in calls:
        assert "train_qwen" not in cmd[1]
        checkpoint = cmd[cmd.index("--model-path") + 1] if "--model-path" in cmd else cmd[cmd.index("--adapter") + 1]
        assert checkpoint.endswith("checkpoint_epoch_002.pt")
        out = cmd[cmd.index("--output-dir") + 1] if "--output-dir" in cmd else cmd[cmd.index("--output-root") + 1]
        assert "selected_epoch_002" in out
    selected = tmp_path / "selected_epoch_002/cirr_val"
    selected.mkdir(parents=True)
    (selected / "val_metrics.json").write_text('{"recall_at_1": 12}')
    suite.summarize()
    assert json.loads((tmp_path / "all_metrics.json").read_text())[0]["CIRR/recall_at_1"] == 12


@pytest.mark.parametrize("name", ["oprd_s2", "random_kl_s2", "infonce", "cmse_s2"])
def test_direct_configs_match_clip_except_oom_batch_adjustment(name):
    clip = load_on_policy_config(PROJECT / "configs/main9_clip_l_infonce.yaml")
    qwen = load_qwen_on_policy_config(PROJECT / f"configs/qwen2_shared_direct_{name}.yaml")
    # Original CLIP-sized execution OOMed; retain effective batch 32 as requested.
    assert qwen.optimizer.batch_size_per_gpu == 16
    assert qwen.optimizer.gradient_accumulation_steps == 2
    assert qwen.optimizer.candidate_encode_batch_size == 32
    assert (qwen.optimizer.batch_size_per_gpu * qwen.optimizer.gradient_accumulation_steps
            == clip.optimizer.batch_size_per_gpu * clip.optimizer.gradient_accumulation_steps == 32)
    assert qwen.model.gradient_checkpointing == clip.model.gradient_checkpointing
    assert not qwen.optimizer.candidate_gradient_cache
    assert qwen.mining.image_batch_size == clip.mining.image_batch_size == 512
    assert qwen.mining.query_batch_size == 128  # lower CPU preprocessing memory during mining
    assert qwen.data.num_workers == clip.data.num_workers == 8
    assert qwen.data.prefetch_factor == clip.data.prefetch_factor == 2
    assert qwen.evaluation.batch_size == clip.evaluation.batch_size == 256
    assert qwen.optimizer.epochs == clip.optimizer.epochs == 4
    assert qwen.optimizer.weight_decay == clip.optimizer.weight_decay
    assert qwen.optimizer.warmup_ratio == clip.optimizer.warmup_ratio
    assert qwen.optimizer.max_grad_norm == clip.optimizer.max_grad_norm
    assert qwen.optimizer.mixed_precision == clip.optimizer.mixed_precision
    assert qwen.mining.top_k == clip.mining.top_k == 32
    assert qwen.mining.force_target_insertion and not qwen.objective.uses_teacher_filter
    assert not qwen.mining.live_reservoir_selection
    assert qwen.model.lora_target_modules == "all-linear"
    assert qwen.optimizer.learning_rate == 2e-5  # retained agreed LoRA-specific LR
    assert "qwen_shared_lora_direct_e4" in str(qwen.experiment.output_dir)


def test_direct_forward_runs_each_encoder_once_with_live_gradients():
    class Student(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.device = torch.device("cpu")
            self.encoder = torch.nn.Linear(4, 3)
            self.calls = []
        def encode(self, x, branch, requires_grad):
            assert requires_grad and torch.is_grad_enabled()
            self.calls.append(branch)
            return torch.nn.functional.normalize(self.encoder(x), dim=-1)
        def encode_prepared_queries(self, x, *, requires_grad):
            return self.encode(x, "query", requires_grad)
        def encode_prepared_images(self, x, *, requires_grad):
            return self.encode(x, "image", requires_grad)
    s = Student()
    q = torch.randn(2, 4, requires_grad=True)
    a, b = torch.randn(2, 4, requires_grad=True), torch.randn(1, 4, requires_grad=True)
    scores = compute_live_qwen_scores(s, {"queries": q, "images": [a, b],
                                         "candidate_positions": torch.tensor([[0, 1, 2], [2, 0, 1]])})
    scores[:, 0].sum().backward()
    assert s.calls == ["query", "image", "image"]
    assert q.grad.abs().sum() > 0 and a.grad.abs().sum() > 0 and b.grad.abs().sum() > 0


@pytest.mark.parametrize("epochs", [2, 4])
@pytest.mark.parametrize("clip_style", [False, True])
def test_clip_style_runner_evaluates_all_epochs_and_keeps_final_paths(tmp_path, monkeypatch, epochs, clip_style):
    spec = importlib.util.spec_from_file_location("qwen_direct_suite_test", PROJECT / "scripts/run_main_comparison_e4.py")
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)
    c = load_qwen_on_policy_config(PROJECT / "configs/qwen2_shared_direct_oprd_s2.yaml")
    c.experiment.output_dir = tmp_path
    c.optimizer.epochs = epochs
    name = "qwen2_shared_direct_oprd_s2"
    monkeypatch.setattr(suite, "load", lambda _: c)
    monkeypatch.setitem(suite.runner.CONFIG_PATHS, name, PROJECT / f"configs/{name}.yaml")
    monkeypatch.setattr(suite.runner, "LABEL_PREFIX", "SHARED-DIRECT")
    monkeypatch.delenv("QWEN2_EVAL_BATCH_SIZE", raising=False)
    monkeypatch.delenv("QWEN2_DIAG_BATCH_SIZE", raising=False)
    for epoch in range(1, epochs + 1):
        (tmp_path / f"checkpoint_epoch_{epoch:03d}.pt").touch()
    commands = []
    def execute(command, *args, **kwargs):
        command = list(map(str, command))
        commands.append(command)
        target = Path(command[command.index("--output-dir") + 1]) if "--output-dir" in command else Path(command[command.index("--output-root") + 1]) / f"SHARED-DIRECT-{name}"
        filename = ("val_metrics.json" if "evaluate_cirr" in command[1] else
                    "metrics.json" if "evaluate_fashioniq" in command[1] else
                    "official_metrics.csv" if "pinpoint" in command[1] else "summary.csv")
        target.mkdir(parents=True, exist_ok=True)
        (target / filename).touch()
    monkeypatch.setattr(suite.runner, "execute", execute)
    suite.run_qwen(name, clip_style=clip_style)
    retrieval_count = 2 * epochs if clip_style else 2
    assert len(commands) == retrieval_count + 2
    for command in commands[:retrieval_count]:
        assert command[command.index("--image-batch-size") + 1] == ("256" if clip_style else "16")
        assert command[command.index("--query-batch-size") + 1] == ("256" if clip_style else "16")
    for command in commands[retrieval_count:]:
        assert command[command.index("--batch-size") + 1] == ("128" if clip_style else "16")
        assert f"checkpoint_epoch_{epochs:03d}.pt" in command[command.index("--adapter") + 1]
    assert (tmp_path / "cirr_val/epoch_001/val_metrics.json").exists() == clip_style
    assert (tmp_path / "cirr_val/val_metrics.json").exists()
    assert (tmp_path / f"fashioniq/epoch_{epochs - 1:03d}/metrics.json").exists() == clip_style
    assert (tmp_path / "fashioniq/metrics.json").exists()
    suite.run_qwen(name, clip_style=clip_style)
    assert len(commands) == retrieval_count + 2  # completed evaluations are not repeated
