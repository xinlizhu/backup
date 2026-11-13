import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("transformers.models.qwen3_5")
from transformers import AutoProcessor, Qwen3_5Config, Qwen3_5ForConditionalGeneration, Qwen3_5Model

from cir_qdiag_train import qwen35_student as implementation
from cir_qdiag_train.qwen_on_policy_config import load_qwen_on_policy_config
from cir_qdiag_train.qwen_student import load_qwen_checkpoint, qwen_student_from_checkpoint, save_qwen_checkpoint

PROJECT = Path(__file__).parents[1]


def config(mode):
    return load_qwen_on_policy_config(PROJECT / f"configs/qwen35_{mode}_oprd_s2.yaml")


def tiny_config():
    return Qwen3_5Config(
        text_config={"vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
                     "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2,
                     "head_dim": 8, "linear_key_head_dim": 4, "linear_value_head_dim": 4,
                     "linear_num_key_heads": 2, "linear_num_value_heads": 4,
                     "layer_types": ["linear_attention"] * 3 + ["full_attention"],
                     "rope_parameters": {"rope_type": "default", "mrope_section": [1, 1, 2],
                                         "partial_rotary_factor": 1.0}},
        vision_config={"depth": 2, "hidden_size": 32, "intermediate_size": 64,
                       "num_heads": 4, "out_hidden_size": 32, "num_position_embeddings": 16,
                       "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2},
        image_token_id=10, video_token_id=11, vision_start_token_id=12, vision_end_token_id=13,
    )


def inputs():
    return {"input_ids": torch.tensor([[12, 10, 13, 5, 6]]),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
            "mm_token_type_ids": torch.tensor([[0, 1, 0, 0, 0]]),
            "pixel_values": torch.randn(4, 3 * 2 * 16 * 16),
            "image_grid_thw": torch.tensor([[1, 2, 2]])}


def student(mode, monkeypatch):
    base = Qwen3_5Model(tiny_config())
    monkeypatch.setattr(implementation, "load_qwen35_backbone", lambda *args: deepcopy(base))
    monkeypatch.setattr(AutoProcessor, "from_pretrained", lambda *args, **kwargs: None)
    c = config(mode)
    c.model.lora_rank, c.model.lora_alpha = 2, 4
    return implementation.Qwen35Student(c.model, device=torch.device("cpu"), mixed_precision="no"), c


def test_paired_configs_only_trainability_differs():
    lora, full = config("lora"), config("full")
    a, b = lora.model_dump(mode="json"), full.model_dump(mode="json")
    for item in [a, b]:
        item["experiment"].pop("name")
        item["experiment"].pop("output_dir")
        item["model"].pop("variant")
    assert a == b
    for c in [lora, full]:
        assert c.model.train_gallery and c.data.student_gallery_cache_dir is None
        assert c.optimizer.batch_size_per_gpu == 16
        assert c.optimizer.gradient_accumulation_steps == 2
        assert c.optimizer.batch_size_per_gpu * c.optimizer.gradient_accumulation_steps == 32
        assert c.data.num_workers == c.evaluation.num_workers == 8
        assert c.data.prefetch_factor == c.evaluation.prefetch_factor == 2
        assert c.mining.top_k == 32 and c.mining.force_target_insertion
        assert not c.objective.uses_teacher_filter
        assert not c.optimizer.candidate_gradient_cache


@pytest.mark.parametrize("mode", ["lora", "full"])
def test_clip_batch_configs_match_clip_without_changing_other_settings(mode):
    from cir_qdiag_train.on_policy_config import load_on_policy_config

    previous = config(mode)
    name = f"qwen35_clip_batch_{mode}_oprd_s2"
    current = load_qwen_on_policy_config(PROJECT / f"configs/{name}.yaml")
    clip = load_on_policy_config(PROJECT / "configs/epoch_control_kl_all_inject.yaml")
    assert current.optimizer.batch_size_per_gpu == clip.optimizer.batch_size_per_gpu == 32
    assert current.optimizer.gradient_accumulation_steps == clip.optimizer.gradient_accumulation_steps == 1
    assert current.optimizer.candidate_encode_batch_size == clip.optimizer.live_image_batch_size == 256
    assert current.data.num_workers == clip.data.num_workers == 8
    assert current.data.prefetch_factor == clip.data.prefetch_factor == 2
    assert current.experiment.name == name
    assert current.experiment.output_dir == PROJECT / "outputs/qwen35_clip_batch_e4" / name
    # No old checkpoints, mining outputs or live-run configuration are modified.
    assert current.experiment.output_dir != previous.experiment.output_dir
    assert previous.optimizer.batch_size_per_gpu == 16
    assert previous.optimizer.gradient_accumulation_steps == 2
    assert previous.optimizer.candidate_encode_batch_size == 16
    a, b = previous.model_dump(mode="json"), current.model_dump(mode="json")
    for item in [a, b]:
        item["experiment"].pop("name")
        item["experiment"].pop("output_dir")
        for key in ["batch_size_per_gpu", "gradient_accumulation_steps", "candidate_encode_batch_size"]:
            item["optimizer"].pop(key)
    assert a == b


@pytest.mark.parametrize("index,mode", [(0, "lora"), (1, "full")])
@pytest.mark.parametrize("clip_batch", [False, True])
def test_qwen35_entrypoints_select_only_requested_run(index, mode, clip_batch):
    import subprocess
    import sys

    prefix = "qwen35_clip_batch" if clip_batch else "qwen35"
    result = subprocess.run(
        [sys.executable, str(PROJECT / f"scripts/run_{prefix}_e4.py"),
         "--run-indices", str(index), "--dry-run"],
        cwd=PROJECT, capture_output=True, text=True, check=True, timeout=30,
    )
    assert f"[{index}] {prefix}_{mode}_oprd_s2" in result.stdout
    assert f"[{1 - index}]" not in result.stdout
    assert "workers=8; prefetch=2" in result.stdout
    assert ("batch=32 x 1; candidate_chunk=256" if clip_batch else
            "batch=16 x 2; candidate_chunk=16") in result.stdout
    assert f"outputs/{prefix}_e4/{prefix}_{mode}_oprd_s2" in result.stdout


@pytest.mark.parametrize("mode", ["lora", "full"])
def test_both_sides_and_all_components_have_gradients(mode, monkeypatch):
    s, _ = student(mode, monkeypatch)
    s.trainability_audit()
    for encode in [s.encode_prepared_queries, s.encode_prepared_images]:
        s.train()
        s.model.zero_grad(set_to_none=True)
        x = inputs()
        x["pixel_values"].requires_grad_(True)
        result = encode(x, requires_grad=True)
        (result * torch.arange(32)).sum().backward()
        assert x["pixel_values"].grad.abs().sum() > 0
        for fragment in ["linear_attn", "self_attn", "visual.blocks", "visual.merger", ".mlp."]:
            gradients = [p.grad for n, p in s.model.named_parameters() if fragment in n and p.requires_grad]
            assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
            assert any(g.abs().sum() > 0 for g in gradients)
        if mode == "full":
            assert all(p.requires_grad and p.grad is not None for p in s.model.parameters())
        else:
            assert all(p.grad is None for p in s.model.parameters() if not p.requires_grad)


@pytest.mark.parametrize("mode", ["lora", "full"])
def test_checkpoint_eval_and_optimizer_resume(mode, monkeypatch, tmp_path):
    s, c = student(mode, monkeypatch)
    c.experiment.output_dir = tmp_path
    s.train()
    x = inputs()
    opt = torch.optim.AdamW(s.trainable_parameters(), lr=2e-5)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
    (s.encode_prepared_images(x, requires_grad=True) * torch.arange(32)).sum().backward()
    opt.step()
    sched.step()
    s.eval()
    expected = s.encode_prepared_images(x)
    path = tmp_path / "last.pt"
    save_qwen_checkpoint(path, student=s, optimizer=opt, scheduler=sched, epoch=0,
                         global_step=1, config=c.model_dump(mode="json"))
    rng = torch.get_rng_state().clone()
    restored, payload = qwen_student_from_checkpoint(path, device=torch.device("cpu"), mixed_precision="no")
    torch.testing.assert_close(expected, restored.encode_prepared_images(x))
    restored_opt = torch.optim.AdamW(restored.trainable_parameters(), lr=1e-3)
    restored_sched = torch.optim.lr_scheduler.StepLR(restored_opt, step_size=1)
    load_qwen_checkpoint(path, student=restored, optimizer=restored_opt, scheduler=restored_sched,
                         expected_config=c.model_dump(mode="json"))
    assert restored_opt.param_groups[0]["lr"] == opt.param_groups[0]["lr"]
    assert restored_sched.state_dict() == sched.state_dict()
    assert torch.equal(torch.get_rng_state(), rng)
    wrong = dict(payload, weight_format="lora_adapter" if mode == "full" else "full_backbone")
    with pytest.raises(ValueError, match="not interchangeable"):
        restored.load_checkpoint_weights(wrong)


def test_native_loading_preserves_backbone_weights(tmp_path):
    reference = Qwen3_5ForConditionalGeneration(tiny_config())
    reference.save_pretrained(tmp_path)
    c = config("full").model
    c.pretrained_model_name_or_path = str(tmp_path)
    loaded = implementation.load_qwen35_backbone(c, torch.float32)
    for name, value in reference.model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)


def test_real_model_structure_and_preprocessing(monkeypatch):
    from PIL import Image
    from cir_qdiag_train.qwen_student import _composed_conversation
    from cir_qdiag_train.qwen_training_data import ImageCollator, QueryCollator

    c = config("lora").model
    processor = AutoProcessor.from_pretrained(c.pretrained_model_name_or_path, local_files_only=True)
    im = Image.new("RGB", (64, 96), (20, 50, 100))
    conv = [_composed_conversation([im], "make it red", instruction="Represent the composed image retrieval query.",
                                  min_pixels=c.min_pixels, max_pixels=c.max_pixels)]
    expected = implementation.preprocess_qwen35(processor, conv, c.max_length)
    actual = QueryCollator(processor, c)([(0, im, "make it red")])[1]
    for k, v in expected.items():
        torch.testing.assert_close(actual[k], v, rtol=0, atol=0)
    assert actual["mm_token_type_ids"].eq(1).any()
    assert ImageCollator(processor, c)([(0, im)])[1]["mm_token_type_ids"].eq(1).any()
    with torch.device("meta"):
        base = Qwen3_5Model(Qwen3_5Config.from_pretrained(c.pretrained_model_name_or_path, local_files_only=True))
    assert sum(p.numel() for p in base.parameters()) == 852985920
    monkeypatch.setattr(implementation, "load_qwen35_backbone", lambda *args: base)
    monkeypatch.setattr(AutoProcessor, "from_pretrained", lambda *args, **kwargs: None)
    with torch.device("meta"):
        s = implementation.Qwen35Student(c, device=torch.device("meta"), mixed_precision="no")
    assert s.parameter_summary()["trainable_parameter_count"] == 13345792
    assert sum(len(v) for v in s.trainability_audit()["modules"].values()) == 236


@pytest.mark.parametrize("mode", ["lora", "full"])
@pytest.mark.parametrize("loss_type", ["rank_kd", "candidate_infonce", "centered_mse"])
def test_training_loop_uses_live_candidates_for_both_modes(mode, loss_type, monkeypatch, tmp_path):
    from cir_qdiag_train import qwen_on_policy_trainer as trainer

    c = config(mode)
    c.experiment.output_dir = tmp_path
    c.data.num_workers = 0
    c.optimizer.epochs = 1
    c.optimizer.batch_size_per_gpu = 2
    c.optimizer.gradient_accumulation_steps = 2
    c.optimizer.mixed_precision = "no"
    c.objective.loss_type = loss_type
    if loss_type != "rank_kd":
        c.mining.candidate_source = "random"
        c.objective.rank_weight = 0
    records = [SimpleNamespace(reference_id="0", target_id=str(i), reference_path="unused", text="edit")
               for i in [1, 2, 3, 4]]
    shared_calls = []
    gallery_refreshes = []

    class Student(torch.nn.Module):
        def __init__(self, config, **kwargs):
            super().__init__()
            self.config = config
            self.device = torch.device("cpu")
            self.weight = torch.nn.Parameter(torch.eye(4))

        def trainable_parameters(self):
            return [self.weight]

        def parameter_summary(self):
            return {}

        def trainability_audit(self):
            return {"mode": mode}

        def prepare_query_batch(self, paths, texts):
            return torch.ones(len(paths), 4)

        def encode_prepared_queries(self, x, requires_grad):
            return F_normalize(x @ self.weight)

        def encode_prepared_images(self, x, requires_grad=False):
            shared_calls.append(requires_grad)
            return F_normalize(x @ self.weight)

    def F_normalize(x):
        return torch.nn.functional.normalize(x, dim=-1)

    saved = []
    def save(path, **kwargs):
        saved.append(kwargs["student"].weight.detach().clone())
        path.touch()

    def current_gallery(config, student, paths, epoch):
        gallery_refreshes.append(epoch)
        return np.random.default_rng(42).normal(size=(40, 4)).astype(np.float32)

    def live_batches(**kwargs):
        for indices in [kwargs["order"][:2], kwargs["order"][2:]]:
            yield indices, {"queries": torch.randn(2, 4), "images": [torch.randn(40, 4)],
                            "candidate_positions": torch.from_numpy(np.array(kwargs["candidates"][indices]))}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(implementation, "Qwen35Student", Student)
    monkeypatch.setattr(trainer, "load_image_map", lambda *args: ([str(i) for i in range(40)], ["unused"] * 40))
    monkeypatch.setattr(trainer, "load_cirr_positive_records", lambda *args: records)
    monkeypatch.setattr(trainer, "encode_current_gallery", current_gallery)
    monkeypatch.setattr(trainer, "shared_training_batches", live_batches)
    monkeypatch.setattr(trainer, "save_qwen_checkpoint", save)
    monkeypatch.setattr(trainer, "_load_student_gallery", lambda *args: pytest.fail("immutable gallery used"))
    monkeypatch.setattr(trainer, "_score_teacher_candidates", lambda **kwargs:
                        (np.zeros((4, 32), dtype=np.float32), np.ones(4, dtype=bool)))
    assert trainer.train_qwen_on_policy_distillation(c).is_file()
    assert shared_calls == [True, True]
    assert gallery_refreshes == ([1] if loss_type == "rank_kd" else [])
    assert len(saved) == 2 and not torch.equal(saved[0], torch.eye(4))
    meta = json.loads((tmp_path / "on_policy/epoch_001/metadata.json").read_text())
    assert meta["student_variant"] == c.model.variant


@pytest.mark.parametrize("mode", ["lora", "full"])
@pytest.mark.parametrize("clip_batch", [False, True])
def test_runner_evaluates_final_checkpoint_and_resumes(tmp_path, monkeypatch, mode, clip_batch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("qwen35_suite_test", PROJECT / "scripts/run_main_comparison_e4.py")
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)
    name = f"qwen35_{'clip_batch_' if clip_batch else ''}{mode}_oprd_s2"
    c = load_qwen_on_policy_config(PROJECT / f"configs/{name}.yaml")
    c.experiment.output_dir = tmp_path
    monkeypatch.setattr(suite, "load", lambda _: c)
    monkeypatch.setitem(suite.runner.CONFIG_PATHS, name, PROJECT / f"configs/{name}.yaml")
    monkeypatch.setattr(suite.runner, "LABEL_PREFIX", "QWEN35")
    monkeypatch.setenv("QWEN2_EVAL_BATCH_SIZE", "16")
    monkeypatch.setenv("QWEN2_DIAG_BATCH_SIZE", "16")
    (tmp_path / "last.pt").touch()
    commands = []

    def execute(command, *args, **kwargs):
        command = list(map(str, command))
        commands.append(command)
        if "train_qwen" in command[1]:
            assert "--resume-from" in command
            (tmp_path / "checkpoint_epoch_004.pt").touch()
            return
        directory = (Path(command[command.index("--output-dir") + 1]) if "--output-dir" in command else
                     Path(command[command.index("--output-root") + 1]) / f"QWEN35-{name}")
        directory.mkdir(parents=True, exist_ok=True)
        filename = ("val_metrics.json" if "evaluate_cirr" in command[1] else
                    "metrics.json" if "evaluate_fashioniq" in command[1] else
                    "official_metrics.csv" if "pinpoint" in command[1] else "summary.csv")
        (directory / filename).touch()

    monkeypatch.setattr(suite.runner, "execute", execute)
    suite.run_qwen(name)
    assert len(commands) == 5
    suite.run_qwen(name)
    assert len(commands) == 5
