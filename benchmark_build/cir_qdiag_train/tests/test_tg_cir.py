"""CPU tests: TG-CIR formula parity, padding, loaders, resume, evaluation, registry."""

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch.nn import functional as F
from transformers import CLIPConfig, CLIPModel

from cir_qdiag_train.tg_cir_config import (
    DEFAULT_CONFIG,
    PROJECT,
    clip_backbone_name,
    load_config,
    model_label,
    training_identity,
)
from cir_qdiag_train.tg_cir_data import TGCIRProcessor, build_train_loader, epoch_indices
from cir_qdiag_train.tg_cir_evaluation import score_cirr
from cir_qdiag_train.tg_cir_model import TGCIR, total_loss
from cir_qdiag_train.tg_cir_runtime import (
    build_model,
    evaluation_identity,
    link_checkpoint,
    load_checkpoint,
    restore_rng,
    rng_state,
    save_checkpoint,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_model():
    config = CLIPConfig(
        projection_dim=8,
        text_config={
            "vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "max_position_embeddings": 77,
            "eos_token_id": 31,
        },
        vision_config={
            "hidden_size": 12,
            "intermediate_size": 24,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "image_size": 16,
            "patch_size": 8,
        },
    )
    config._attn_implementation = "eager"
    return TGCIR(CLIPModel(config), global_tokens=2, local_tokens=2)


def reference_losses(model, ref, text, target):
    """Literal released loss equations, separately assembled for parity testing."""
    s = model.s_remain_map(torch.cat([ref, text], dim=-1))
    tr = model.t_remain_map(torch.cat([ref.detach(), target], dim=-1))
    tt = model.t_replace_map(torch.cat([text.detach(), target], dim=-1))
    student = s * ref + (1 - s) * text
    teacher = tr * ref.detach() + tt * text.detach()
    weight = model.local_weight.unsqueeze(0).unsqueeze(-1)
    tf = (F.normalize(teacher, p=2, dim=-1) * weight).flatten(1)
    gf = (F.normalize(target, p=2, dim=-1) * weight).flatten(1)
    sq = F.normalize(torch.mean(student, dim=1), p=2, dim=-1)
    sg = F.normalize(torch.mean(target, dim=1), p=2, dim=-1)
    labels = torch.arange(len(ref))
    teacher_logits = F.normalize(gf, dim=-1) @ F.normalize(gf, dim=-1).T / model.temperature
    # Explicitly test upstream Variable behavior, rather than assume it.
    teacher_probs = F.softmax(torch.autograd.Variable(teacher_logits), dim=1)
    assert not teacher_probs.requires_grad
    ortho = []
    for x in (ref, target, text):
        x = F.normalize(x, dim=-1)
        ortho.append(F.mse_loss(x @ x.transpose(1, 2), torch.eye(x.shape[1]).repeat(len(x), 1, 1)))
    return {
        "stu_rank": F.cross_entropy((sq @ sg.T) * model.loss_T, labels),
        "tea_rank": F.cross_entropy((tf @ gf.T) * model.loss_T, labels),
        "tea_mask": F.mse_loss(tr + tt, torch.ones_like(tr)),
        "ortho": sum(ortho) / 3,
        "ckd": F.mse_loss(s, tr.detach()) + F.mse_loss(1 - s, tt.detach()),
        "kl": F.kl_div(
            F.log_softmax((sq @ sg.T) / model.temperature, dim=1),
            teacher_probs,
            reduction="batchmean",
        ),
    }


def test_six_losses_and_gradients_match_released_equations():
    torch.manual_seed(4)
    model = tiny_model()
    reference = copy.deepcopy(model)
    inputs = [torch.randn(3, 4, 8, requires_grad=True) for _ in range(3)]
    copies = [x.detach().clone().requires_grad_() for x in inputs]
    actual = model.losses_from_attributes(*inputs)
    expected = reference_losses(reference, *copies)
    weights = load_config(PROJECT / "configs/tg_cir_clip_b_e4.yaml")["loss_weights"]
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key])
    total_loss(actual, weights).backward()
    total_loss(expected, weights).backward()
    for x, y in zip(inputs, copies, strict=True):
        torch.testing.assert_close(x.grad, y.grad)
    for (_, x), (_, y) in zip(model.named_parameters(), reference.named_parameters(), strict=True):
        if x.grad is not None:
            torch.testing.assert_close(x.grad, y.grad)


def test_detachment_and_inference_has_no_teacher():
    model = tiny_model()
    ref, text, target = [torch.randn(2, 4, 8, requires_grad=True) for _ in range(3)]
    losses = model.losses_from_attributes(ref, text, target)
    losses["tea_rank"].backward(retain_graph=True)
    assert ref.grad is None and text.grad is None
    assert target.grad is not None
    model.zero_grad(set_to_none=True)
    losses["ckd"].backward()
    assert all(p.grad is None for p in model.t_remain_map.parameters())
    model.eval()
    pixels, ids = torch.randn(2, 3, 16, 16), torch.tensor([[1, 4, 31, 0], [1, 6, 31, 0]])
    before = model.extract_retrieval_compose(pixels, ids)
    with torch.no_grad():
        for module in (model.t_remain_map, model.t_replace_map):
            for parameter in module.parameters():
                parameter.fill_(100)
    after = model.extract_retrieval_compose(pixels, ids)
    torch.testing.assert_close(before, after)
    torch.testing.assert_close(after.norm(dim=-1), torch.ones(2))


def test_backbone_training_step_and_attribute_shapes():
    model = tiny_model()
    pixels = torch.randn(2, 3, 16, 16)
    ids = torch.tensor([[1, 4, 31, 0], [1, 6, 31, 0]])
    image_attrs = model.backbone.extract_img_fea(pixels)
    text_attrs = model.backbone.extract_text_fea(ids)
    assert image_attrs.shape == text_attrs.shape == (2, 4, 8)
    assert model.backbone.masks.weight[0, 0] == 1
    assert model.backbone.masks.weight[0, -1].item() == pytest.approx(0.1)
    model.backbone.clip.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    losses = model(pixels, ids, pixels.flip(0))
    sum(losses.values()).backward()
    assert model.backbone.clip.vision_model.embeddings.patch_embedding.weight.grad is not None
    assert model.backbone.clip.text_model.embeddings.token_embedding.weight.grad is not None
    assert all(torch.isfinite(value) for value in losses.values())


def test_sampler_resume_and_drop_last():
    full = epoch_indices(103, 32, 42, 1)
    assert len(full) == 96 and len(set(full)) == 96
    assert epoch_indices(103, 32, 42, 1, 1) == full[32:]
    assert epoch_indices(103, 32, 42, 1, 3) == []
    assert full != epoch_indices(103, 32, 42, 2)
    with pytest.raises(ValueError):
        epoch_indices(31, 32, 42, 1)


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        assert kwargs["max_length"] == 77 and not kwargs["padding"]
        return {"input_ids": [[30, 4, 31] for _ in texts]}


class SmallProcessor:
    def __init__(self):
        from torchvision.transforms import ToTensor

        self.transform = ToTensor()

    def tokenize(self, texts):
        return torch.tensor([[1, 4, 31, 0] for _ in texts])


def test_zero_padding():
    processor = TGCIRProcessor.__new__(TGCIRProcessor)
    processor.tokenizer = FakeTokenizer()
    ids = processor.tokenize(["one", "two"])
    assert ids.shape == (2, 77)
    assert not ids[:, 3:].any()
    assert ids.argmax(dim=-1).tolist() == [2, 2]


@pytest.mark.parametrize("workers", [0, 2])
def test_cpu_only_training_workers(tmp_path, workers):
    path = tmp_path / "image.png"
    Image.new("RGB", (16, 16)).save(path)
    records = [SimpleNamespace(reference_path=path, target_path=path, text="red") for _ in range(8)]
    config = {
        "seed": 42,
        "training": {"batch_size": 2, "num_workers": workers, "prefetch_factor": 2},
    }
    loader = build_train_loader(records, SmallProcessor(), config, 1, start_step=2)
    batches = list(loader)
    assert len(batches) == 2
    assert batches[0]["reference_pixels"].shape == (2, 3, 16, 16)
    assert all(v.device.type == "cpu" for batch in batches for v in batch.values())


def test_checkpoint_aliases_are_immutable_and_rng_restores(tmp_path):
    last, epoch = tmp_path / "last.pt", tmp_path / "epoch.pt"
    before = rng_state()
    expected = torch.rand(3)
    restore_rng(before)
    torch.testing.assert_close(torch.rand(3), expected)
    save_checkpoint(last, {"method": "tg_cir_v1", "value": torch.tensor(1)})
    link_checkpoint(last, epoch)
    save_checkpoint(last, {"method": "tg_cir_v1", "value": torch.tensor(2)})
    assert load_checkpoint(epoch)["value"] == 1
    assert load_checkpoint(last)["value"] == 2


def test_config_and_cache_identity(tmp_path):
    config = load_config(PROJECT / "configs/tg_cir_clip_b_e4.yaml")
    assert config["training"]["epochs"] == 4
    assert config["training"]["batch_size"] == 32
    assert config["training"]["num_workers"] == 8
    assert config["training"]["prefetch_factor"] == 2
    altered = copy.deepcopy(config)
    altered["training"]["num_workers"] = 0
    assert training_identity(config) == training_identity(altered)
    altered["model"]["temperature"] = 1
    assert training_identity(config) != training_identity(altered)
    checkpoint, folder = tmp_path / "last.pt", tmp_path / "eval"
    save_checkpoint(checkpoint, {"method": "tg_cir_v1"})
    evaluation_identity(folder, checkpoint, config)
    evaluation_identity(folder, checkpoint, config)
    with pytest.raises(ValueError, match="changed"):
        evaluation_identity(folder, checkpoint, altered)


def test_cirr_masks_reference_and_reports_all_seven_metrics():
    records = [SimpleNamespace(reference_id="ref", target_id="target")]
    metrics = score_cirr(
        np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32),
        np.array([[1.0, 0.0]], dtype=np.float32),
        ["ref", "target", "other"],
        records,
        [{"img_set": {"members": ["ref", "target", "other"]}}],
        torch.device("cpu"),
    )
    assert len(metrics) == 8
    assert all(v == 100 for k, v in metrics.items() if "recall" in k)


def test_registry_and_single_reference_adapter_equivalence():
    workspace = PROJECT.parents[1]
    sys.path.insert(0, str(workspace / "CIR_PLAY"))
    from run_pinpoint_bge_lincir import infer_adapter_kind

    from cir_qdiag_train.tg_cir_adapter import TGCIRAdapter

    assert infer_adapter_kind("TG-CIR-CLIP-B-E4", Path("checkpoint.pt")) == "tg_cir"
    adapter = TGCIRAdapter.__new__(TGCIRAdapter)
    adapter.device, adapter.amp_dtype = torch.device("cpu"), torch.float32
    adapter.model = tiny_model().eval()
    adapter.processor = SmallProcessor()
    pixels = torch.randn(3, 3, 16, 16)
    ids = torch.tensor([[1, 4, 31, 0], [1, 4, 31, 0]])
    prepared = {"pixel_values": pixels, "input_ids": ids, "reference_counts": [1, 2]}
    actual = adapter.encode_prepared_query_batch(prepared, Path("/"))
    expected = adapter.model.extract_retrieval_compose(pixels[:1], ids[:1]).detach().numpy()
    np.testing.assert_allclose(actual[:1], expected, atol=1e-6)
    assert actual.shape == (2, 8)
    rows = pd.DataFrame({"instruction": ["red"], "query_image_signature": ["ref"]})
    with pytest.raises(ValueError, match="counts"):
        adapter.prepare_query_batch(rows, Path("/"), [])


@pytest.mark.parametrize("variant", ["b", "l"])
def test_runner_only_uses_final_epoch_and_adapter_flag(variant):
    spec = importlib.util.spec_from_file_location("tg_runner", PROJECT / "scripts/run_tg_cir_e4.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config_path = PROJECT / f"configs/tg_cir_clip_{variant}_e4.yaml"
    config = load_config(config_path)
    cmds = list(module.commands(config_path, config, "cuda:0"))
    assert len(cmds) == 5
    assert all("best.pt" not in str(cmd) for cmd in cmds)
    assert "checkpoint_epoch_004.pt" in str(cmds[-1])
    assert "--adapter" in cmds[-1] and "--adapter" in cmds[-2]
    assert f"TG-CIR-CLIP-{variant.upper()}-E4=" in str(cmds[-1])
    assert f"outputs/tg_cir_clip_{variant}_e4" in str(cmds[-1])
    assert len(list(module.commands(Path("config"), config, "cpu", training_only=True))) == 1


def test_default_is_large_and_model_dimensions_adapt_without_weights():
    config = load_config(DEFAULT_CONFIG)
    assert DEFAULT_CONFIG.name == "tg_cir_clip_l_e4.yaml"
    assert config["training"]["epochs"] == 4 and config["training"]["batch_size"] == 32
    assert config["training"]["num_workers"] == 8 and config["training"]["prefetch_factor"] == 2
    assert model_label(config) == "TG-CIR-CLIP-L"
    path = Path(config["model"]["pretrained_model_name_or_path"]) / "config.json"
    backbone_config = json.loads(path.read_text())
    with torch.device("meta"):
        model = build_model(config, backbone_config)
    assert model.backbone.fc.in_features == 1024
    assert model.backbone.fc.out_features == model.backbone.text_fc.out_features == 768
    assert model.backbone.text_fc.in_features == 768
    assert model.backbone.masks.weight.shape == (4, 768)
    assert model.s_remain_map[0].in_features == 1536
    assert model.t_remain_map[0].in_features == model.t_replace_map[0].in_features == 1536
    backbone_config["vision_config"]["image_size"] = 336
    with pytest.raises(ValueError, match="224px"):
        clip_backbone_name(backbone_config)
    old = load_config(PROJECT / "configs/tg_cir_clip_b_e4.yaml")
    assert training_identity(config) != training_identity(old)


def test_interrupted_training_resumes_identical_optimizer_and_weights(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("tg_trainer", PROJECT / "scripts/train_tg_cir.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    monkeypatch.setattr(trainer, "build_model", lambda config, clip_config=None: tiny_model())
    monkeypatch.setattr(trainer, "TGCIRProcessor", lambda path: SmallProcessor())
    monkeypatch.setattr(trainer, "evaluate_cirr", lambda *args: {"recall_at_1": 10.0})
    image = tmp_path / "toy.png"
    Image.new("RGB", (16, 16), (32, 64, 128)).save(image)
    records = [
        SimpleNamespace(reference_path=image, target_path=image, text="blue") for _ in range(8)
    ]
    config = load_config(PROJECT / "configs/tg_cir_clip_b_e4.yaml")
    config["training"].update(epochs=2, batch_size=2, num_workers=0, save_every_steps=1)
    config["output_dir"] = str(tmp_path / "uninterrupted")
    Path(config["output_dir"]).mkdir()
    args = SimpleNamespace(device="cpu", smoke_test=False)
    trainer.run(config, records, args)
    expected = load_checkpoint(Path(config["output_dir"]) / "last.pt")
    config["output_dir"] = str(tmp_path / "resumed")
    Path(config["output_dir"]).mkdir()
    save = trainer.save_checkpoint

    def save_then_crash(path, payload):
        save(path, payload)
        raise RuntimeError("simulated interruption after a committed step")

    monkeypatch.setattr(trainer, "save_checkpoint", save_then_crash)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        trainer.run(config, records, args)
    assert load_checkpoint(Path(config["output_dir"]) / "last.pt")["next_step"] == 1
    monkeypatch.setattr(trainer, "save_checkpoint", save)
    trainer.run(config, records, args)
    actual = load_checkpoint(Path(config["output_dir"]) / "last.pt")
    assert actual["epoch"] == 2 and actual["next_step"] == 4 and actual["global_step"] == 8
    for name, value in expected["model"].items():
        torch.testing.assert_close(value, actual["model"][name], rtol=0, atol=0)
    for index, state in expected["optimizer"]["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(
                value, actual["optimizer"]["state"][index][key], rtol=0, atol=0
            )
