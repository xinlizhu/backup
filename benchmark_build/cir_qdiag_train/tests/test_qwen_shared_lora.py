from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from cir_qdiag_train.qwen_gradient_cache import shared_student_scores
from cir_qdiag_train.qwen_on_policy_config import QwenQueryLoRAConfig, load_qwen_on_policy_config
from cir_qdiag_train.qwen_student import QwenQueryLoRAStudent

PROJECT = Path(__file__).parents[1]


def test_shared_configs():
    for name in ["oprd_s2", "random_kl_s2", "infonce", "cmse_s2"]:
        c = load_qwen_on_policy_config(PROJECT / f"configs/qwen2_shared_{name}.yaml")
        assert c.model.shared_lora and c.model.lora_target_modules == "all-linear"
        assert c.model.lora_rank == 16 and c.model.lora_alpha == 32
        assert c.data.student_gallery_cache_dir is None
        assert c.mining.top_k == 32 and c.mining.force_target_insertion
        assert not c.objective.uses_teacher_filter
        assert (c.mining.candidate_source == "student") == (name == "oprd_s2")
        assert c.optimizer.batch_size_per_gpu * c.optimizer.gradient_accumulation_steps == 32
    old = load_qwen_on_policy_config(PROJECT / "configs/main9_qwen2_infonce.yaml")
    assert not old.model.shared_lora and old.model.lora_target_modules == "attention"
    with pytest.raises(ValueError, match="all-linear"):
        QwenQueryLoRAConfig(variant="qwen3_vl_shared_lora", pretrained_model_name_or_path="unused")


class ToyStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        self.encoder = torch.nn.Sequential(torch.nn.Linear(5, 8), torch.nn.Dropout(0.3),
                                           torch.nn.Linear(8, 4))

    def encode_prepared_queries(self, x, *, requires_grad):
        return F.normalize(self.encoder(x), dim=-1)

    encode_prepared_images = encode_prepared_queries


@pytest.mark.parametrize("loss_type", ["infonce", "kl", "cmse"])
def test_gradient_cache_matches_direct_with_dropout_and_accumulation(loss_type):
    torch.manual_seed(10)
    direct, cached = ToyStudent(), ToyStudent()
    cached.load_state_dict(direct.state_dict())
    prepared = {"queries": torch.randn(2, 5),
                "images": [torch.randn(2, 5), torch.randn(1, 5)],
                "candidate_positions": torch.tensor([[0, 1, 2], [2, 0, 1]])}
    teacher = torch.randn(2, 3)
    def objective(scores):
        if loss_type == "infonce":
            return F.cross_entropy(scores / 0.07, torch.zeros(2, dtype=torch.long))
        if loss_type == "kl":
            ls, lt = (scores / .05).log_softmax(-1), (teacher / .05).log_softmax(-1)
            return (ls.exp() * (ls - lt)).sum(-1).mean()
        delta = scores - teacher
        return ((delta - delta.mean(-1, keepdim=True)) ** 2).mean()
    for _ in range(2):
        state = torch.get_rng_state()
        s1, replay1 = shared_student_scores(direct, prepared, gradient_cache=False)
        (objective(s1) * 128 / 2).backward()  # simulated AMP scale + accumulation
        replay1()
        expected_rng = torch.get_rng_state()
        torch.set_rng_state(state)
        s2, replay2 = shared_student_scores(cached, prepared, gradient_cache=True)
        (objective(s2) * 128 / 2).backward()
        replay2()
        torch.testing.assert_close(s1, s2)
        assert torch.equal(expected_rng, torch.get_rng_state())
    for p1, p2 in zip(direct.parameters(), cached.parameters()):
        torch.testing.assert_close(p1.grad, p2.grad, rtol=2e-5, atol=2e-4)
    assert sum(p.grad.abs().sum() for p in cached.parameters()) > 0


def tiny_config():
    from transformers import Qwen3VLConfig
    return Qwen3VLConfig(
        text_config={"vocab_size": 32, "hidden_size": 32, "intermediate_size": 64,
                     "num_hidden_layers": 2, "num_attention_heads": 4,
                     "num_key_value_heads": 2, "head_dim": 8,
                     "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]}},
        vision_config={"depth": 2, "hidden_size": 32, "intermediate_size": 64,
                       "num_heads": 4, "out_hidden_size": 32, "num_position_embeddings": 16,
                       "deepstack_visual_indexes": [0], "patch_size": 16,
                       "spatial_merge_size": 2, "temporal_patch_size": 2},
        image_token_id=10, video_token_id=11, vision_start_token_id=12, vision_end_token_id=13,
    )


def build_tiny_student(monkeypatch):
    from transformers import Qwen3VLModel
    from cir_qdiag_train.qwen_retrieval import NativeQwenRetrieval

    template = Qwen3VLModel(tiny_config())
    monkeypatch.setattr(NativeQwenRetrieval, "__init__", lambda self, config, dtype:
                        setattr(self, "model", deepcopy(template)))
    c = QwenQueryLoRAConfig(
        variant="qwen3_vl_shared_lora", pretrained_model_name_or_path=str(PROJECT),
        lora_target_modules="all-linear", lora_rank=2, lora_alpha=4, lora_dropout=0.1,
    )
    return QwenQueryLoRAStudent(c, device=torch.device("cpu"), mixed_precision="no")


def tiny_inputs():
    return {"input_ids": torch.tensor([[12, 10, 13, 5, 6]]),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
            "pixel_values": torch.randn(4, 3 * 2 * 16 * 16),
            "image_grid_thw": torch.tensor([[1, 2, 2]])}


def test_actual_qwen_all_linear_both_branches_checkpoint(monkeypatch, tmp_path):
    from cir_qdiag_train.qwen_on_policy_trainer import audit_shared_adapters
    from cir_qdiag_train.qwen_student import save_qwen_checkpoint, qwen_student_from_checkpoint
    student = build_tiny_student(monkeypatch)
    audit = audit_shared_adapters(student)
    assert all(audit["modules"].values())
    inputs = tiny_inputs()
    for encode in [student.encode_prepared_queries, student.encode_prepared_images]:
        student.model.zero_grad(set_to_none=True)
        student.train()
        emb = encode(inputs, requires_grad=True)
        (emb * torch.arange(32)).sum().backward()
        for component in ["visual.blocks", "merger", "language_model",
                          ".gate_proj", ".up_proj", ".down_proj"]:
            grads = [p.grad for n, p in student.model.named_parameters() if component in n and p.requires_grad]
            assert grads and any(g is not None and g.abs().sum() > 0 for g in grads)
        assert all(p.grad is None for p in student.parameters() if not p.requires_grad)
    student.eval()
    # Nonzero adapter ensures an accidentally disabled image adapter is detectable.
    with torch.no_grad():
        for name, p in student.model.named_parameters():
            if "lora_B" in name:
                p.add_(torch.randn_like(p) * .1)
    a = student.encode_prepared_images(inputs)
    with student.model.disable_adapter():
        b = student.encode_prepared_queries(inputs, requires_grad=False)
    assert not torch.allclose(a, b)
    checkpoint = tmp_path / "shared.pt"
    save_qwen_checkpoint(checkpoint, student=student, optimizer=None, scheduler=None,
                         epoch=0, global_step=1, config={"model": student.config.model_dump(mode="json")})
    # Save/load adapter weights, while retaining the same base initialization.
    payload = torch.load(checkpoint, weights_only=False)
    assert payload["student_type"] == "qwen3_vl_shared_lora"
    assert all("lora_" in name for name in payload["model"])
    student.load_lora_state_dict(payload["model"])
    torch.testing.assert_close(a, student.encode_prepared_images(inputs))
    reloaded, _ = qwen_student_from_checkpoint(checkpoint, device=torch.device("cpu"), mixed_precision="no")
    torch.testing.assert_close(a, reloaded.encode_prepared_images(inputs))
    bad_state = dict(payload["model"])
    bad_state.pop(next(iter(bad_state)))
    with pytest.raises(ValueError, match="Incomplete"):
        reloaded.load_lora_state_dict(bad_state)


def test_instruct_backbone_loading_no_random_weights(monkeypatch, tmp_path):
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
    from cir_qdiag_train.qwen_retrieval import NativeQwenRetrieval
    import accelerate.utils.other
    monkeypatch.setattr(accelerate.utils.other, "is_deepspeed_available", lambda: False)
    model = Qwen3VLForConditionalGeneration(tiny_config())
    model.save_pretrained(tmp_path)
    monkeypatch.setattr(Qwen3VLProcessor, "from_pretrained", lambda *a, **k: None)
    cfg = SimpleNamespace(pretrained_model_name_or_path=str(tmp_path), local_files_only=True)
    loaded = NativeQwenRetrieval(cfg, torch.float32)
    for name, p in model.model.named_parameters():
        torch.testing.assert_close(p, dict(loaded.model.named_parameters())[name])


def test_shared_resume_restores_optimizer_scheduler_rng_and_rejects_changed_loss(monkeypatch, tmp_path):
    from cir_qdiag_train.qwen_student import save_qwen_checkpoint, load_qwen_checkpoint
    student = build_tiny_student(monkeypatch)
    c = load_qwen_on_policy_config(PROJECT / "configs/qwen2_shared_infonce.yaml")
    c.model = student.config
    optimizer = torch.optim.AdamW(student.trainable_parameters(), lr=2e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.5)
    for p in student.trainable_parameters():
        p.grad = torch.ones_like(p)
    optimizer.step()
    scheduler.step()
    saved_weights = deepcopy(student.lora_state_dict())
    path = tmp_path / "resume.pt"
    save_qwen_checkpoint(path, student=student, optimizer=optimizer, scheduler=scheduler,
                         epoch=0, global_step=1, config=c.model_dump(mode="json"))
    expected_random = torch.rand(5)
    optimizer.param_groups[0]["lr"] = 1
    with torch.no_grad():
        for p in student.trainable_parameters():
            p.zero_()
    payload = load_qwen_checkpoint(path, student=student, optimizer=optimizer, scheduler=scheduler,
                                   expected_config=c.model_dump(mode="json"))
    assert payload["global_step"] == 1 and optimizer.param_groups[0]["lr"] == 1e-5
    assert scheduler.last_epoch == 1 and optimizer.state
    torch.testing.assert_close(torch.rand(5), expected_random)
    for name, p in student.lora_state_dict().items():
        torch.testing.assert_close(p, saved_weights[name])
    c.objective.rank_temperature = .3
    with pytest.raises(ValueError, match="configuration mismatch"):
        load_qwen_checkpoint(path, student=student, expected_config=c.model_dump(mode="json"))


def test_shared_random_mining_does_not_encode_gallery(monkeypatch, tmp_path):
    from cir_qdiag_train.qwen_on_policy_trainer import refresh_qwen_candidates
    c = load_qwen_on_policy_config(PROJECT / "configs/qwen2_shared_infonce.yaml")
    c.experiment.output_dir = tmp_path
    path, meta = refresh_qwen_candidates(
        config=c, student=SimpleNamespace(eval=lambda: None),
        records=[SimpleNamespace(reference_id="0", target_id="1")],
        image_ids=[str(i) for i in range(40)], gallery=None, epoch=1, global_step=0,
        device=torch.device("cpu"),
    )
    assert meta["shared_lora"]
    candidates = np.load(path / "candidate_indices.npy")
    assert candidates[0, 0] == 1 and 0 not in candidates


def test_shared_student_mining_refreshes_gallery_per_epoch(monkeypatch, tmp_path):
    import cir_qdiag_train.qwen_on_policy_trainer as trainer
    c = load_qwen_on_policy_config(PROJECT / "configs/qwen2_shared_infonce.yaml")
    c.experiment.output_dir = tmp_path
    c.mining.candidate_source = "student"
    c.mining.top_k = 3
    calls = []
    def gallery(config, student, paths, epoch):
        calls.append(epoch)
        return np.array([[0, 1], [0, 1], [1, 0], [-1, 0]], dtype=np.float32) * (1 if epoch == 1 else -1)
    monkeypatch.setattr(trainer, "encode_current_gallery", gallery)
    monkeypatch.setattr(trainer, "query_batches", lambda **k: [([0], None)])
    student = SimpleNamespace(eval=lambda: None, encode_prepared_queries=lambda *a, **k: torch.tensor([[1., 0.]]))
    kwargs = dict(config=c, student=student, records=[SimpleNamespace(reference_id="0", target_id="1")],
                  image_ids=[str(i) for i in range(4)], image_paths=[Path(str(i)) for i in range(4)],
                  gallery=None, device=torch.device("cpu"))
    p1, _ = trainer.refresh_qwen_candidates(**kwargs, epoch=1, global_step=0)
    trainer.refresh_qwen_candidates(**kwargs, epoch=1, global_step=0)
    p2, _ = trainer.refresh_qwen_candidates(**kwargs, epoch=2, global_step=1)
    assert calls == [1, 2]  # reuse only the exact same epoch's candidate cache
    assert np.load(p1 / "candidate_indices.npy")[0, 1] == 2
    assert np.load(p2 / "candidate_indices.npy")[0, 1] == 3


@pytest.mark.parametrize("gradient_cache", [True, False])
def test_shared_trainer_one_epoch(monkeypatch, tmp_path, gradient_cache):
    import cir_qdiag_train.qwen_on_policy_trainer as trainer
    c = load_qwen_on_policy_config(PROJECT / "configs/qwen2_shared_infonce.yaml")
    c.experiment.output_dir = tmp_path
    c.optimizer.epochs = 1
    c.optimizer.gradient_accumulation_steps = 2
    c.optimizer.mixed_precision = "no"
    c.optimizer.candidate_gradient_cache = gradient_cache
    c.optimizer.log_every_steps = 1
    c.mining.top_k = 3
    c.data.num_workers = 0
    class Student(ToyStudent):
        def __init__(self, *a, **k):
            super().__init__()
        def trainable_parameters(self):
            return list(self.parameters())
        def parameter_summary(self):
            return {}
    records = [SimpleNamespace(reference_id="0", target_id="1") for _ in range(3)]
    saved = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(trainer, "QwenQueryLoRAStudent", Student)
    monkeypatch.setattr(trainer, "audit_shared_adapters", lambda s: {})
    monkeypatch.setattr(trainer, "load_image_map", lambda *a: ([str(i) for i in range(5)], []))
    monkeypatch.setattr(trainer, "load_cirr_positive_records", lambda *a: records)
    def no_frozen_gallery(*a):
        raise AssertionError("shared trainer must not read a fixed Student gallery")
    monkeypatch.setattr(trainer, "_load_student_gallery", no_frozen_gallery)
    def batches(**kwargs):
        for i in kwargs["order"]:
            yield [i], {"queries": torch.randn(1, 5), "images": [torch.randn(2, 5), torch.randn(1, 5)],
                        "candidate_positions": torch.tensor([[0, 1, 2]])}
    monkeypatch.setattr(trainer, "shared_training_batches", batches)
    if not gradient_cache:
        def forbid_gradient_replay(*args, **kwargs):
            raise AssertionError("direct training must not enter the gradient-cache path")
        monkeypatch.setattr(trainer, "shared_student_scores", forbid_gradient_replay)
    def save(path, **kwargs):
        saved.append(kwargs)
        path.touch()
    monkeypatch.setattr(trainer, "save_qwen_checkpoint", save)
    result = trainer.train_qwen_on_policy_distillation(c)
    assert result.is_file() and len(saved) == 2
    assert saved[-1]["global_step"] == 2  # handles the partial last accumulation group
    with pytest.raises(ValueError, match="resume"):
        trainer.train_qwen_on_policy_distillation(c)


class PixelReportingProcessor:
    def apply_chat_template(self, conversations, **kwargs):
        return ["query"] * len(conversations)
    def __call__(self, text, images, **kwargs):
        return {"input_ids": torch.ones(len(images), 4, dtype=torch.long),
                "pixels": torch.tensor([np.asarray(im).mean() for im in images])}


def test_shared_cpu_collator_deduplicates_and_preserves_roles(tmp_path):
    from PIL import Image
    from cir_qdiag_train.qwen_training_data import SharedTrainingCollator
    c = load_qwen_on_policy_config(PROJECT / "configs/qwen2_shared_infonce.yaml")
    paths = []
    for i in range(3):
        p = tmp_path / f"{i}.png"
        Image.new("RGB", (64, 64), (i * 40,) * 3).save(p)
        paths.append(p)
    collator = SharedTrainingCollator(PixelReportingProcessor(), c.model, 2)
    indices, prepared = collator([(4, paths[0], "edit", [paths[1], paths[2]]),
                                  (2, paths[2], "edit", [paths[2], paths[1]])])
    assert indices == [4, 2]
    assert prepared["candidate_positions"].tolist() == [[0, 1], [1, 0]]
    assert prepared["queries"]["pixels"].tolist() == [0, 80]
    assert len(prepared["images"]) == 1
    assert prepared["images"][0]["pixels"].tolist() == [40, 80]
