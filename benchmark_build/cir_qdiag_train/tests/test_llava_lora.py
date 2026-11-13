import importlib.util
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import LlavaNextConfig, LlavaNextForConditionalGeneration, LlavaNextImageProcessor, LlavaNextModel

from cir_qdiag_train import llava_student as implementation
from cir_qdiag_train.qwen_on_policy_config import load_qwen_on_policy_config
from cir_qdiag_train.qwen_student import load_qwen_checkpoint, qwen_student_from_checkpoint, save_qwen_checkpoint
from cir_qdiag_train.qwen_training_data import ImageCollator, QueryCollator, SharedTrainingCollator

PROJECT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
SUFFIXES = ("oprd_s2", "oprd_qwen8", "infonce", "cmse_s2", "cmse_qwen8")


def config(suffix="oprd_s2"):
    return load_qwen_on_policy_config(PROJECT / f"configs/llava_{suffix}.yaml")


def tiny_config():
    return LlavaNextConfig(
        text_config={"model_type": "mistral", "vocab_size": 32064, "hidden_size": 32,
                     "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
                     "num_key_value_heads": 2, "max_position_embeddings": 8192, "sliding_window": None},
        vision_config={"model_type": "clip_vision_model", "hidden_size": 32, "intermediate_size": 64,
                       "num_hidden_layers": 2, "num_attention_heads": 4, "image_size": 32, "patch_size": 16},
        image_grid_pinpoints=[[32, 32]], image_token_index=32000, vision_feature_layer=-2,
        vision_feature_select_strategy="default",
    )


def build_student(monkeypatch, targets="all-linear"):
    c = config()
    c.model.lora_target_modules = targets
    c.optimizer.world_size, c.optimizer.batch_size_per_gpu = 1, 4
    c.model.llava_image_size = 32
    c.model.lora_rank, c.model.lora_alpha = 2, 4
    processor = implementation.load_llava_processor(c.model)
    processor.image_processor = LlavaNextImageProcessor(
        size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32},
        image_grid_pinpoints=[[32, 32]],
    )
    processor.patch_size = 16
    template = LlavaNextModel(tiny_config())
    monkeypatch.setattr(implementation, "load_llava_backbone", lambda *a: deepcopy(template))
    monkeypatch.setattr(implementation, "load_llava_processor", lambda *a: processor)
    return implementation.LlavaLoRAStudent(c.model, device=torch.device("cpu"), mixed_precision="no"), c


@pytest.mark.parametrize("suffix", SUFFIXES)
def test_configs(suffix):
    c = config(suffix)
    assert c.model.llava and c.model.train_gallery and c.model.shared_lora
    assert c.model.lora_target_modules == "attention" and c.model.lora_rank == 16
    assert c.model.lora_alpha == 32
    assert c.data.student_gallery_cache_dir is None
    assert c.optimizer.world_size == 2 and c.optimizer.batch_size_per_gpu == 16
    assert c.optimizer.gradient_accumulation_steps == 1
    assert c.optimizer.candidate_encode_batch_size == 8 and c.optimizer.candidate_gradient_cache
    assert c.data.num_workers == 8
    assert c.data.prefetch_factor == 1
    assert c.evaluation.num_workers == 16
    assert c.evaluation.prefetch_factor == 2
    assert c.optimizer.epochs == 2 and c.optimizer.learning_rate == 5e-6
    assert c.mining.top_k == 32 and c.mining.force_target_insertion
    assert c.mining.candidate_source == ("student" if suffix.startswith("oprd") else "random")
    assert c.objective.rank_temperature == (.07 if suffix == "infonce" else .05)
    assert not c.objective.uses_teacher_filter
    assert c.model.llava_image_size == 512 and c.model.max_length == 8192
    with pytest.raises(ValueError, match="fixed student gallery"):
        type(c).model_validate({**c.model_dump(), "data": {**c.data.model_dump(), "student_gallery_cache_dir": "bad"}})


def test_processor_production_geometry_and_eos():
    c = config().model
    processor = implementation.load_llava_processor(c)
    image = Image.new("RGB", (80, 55), (255, 0, 0))
    prepared = implementation.preprocess_llava(processor, c, [[image], [image, image]], ["red", "add a blue hat"], query=True)
    assert prepared["pixel_values"].shape[:2] == (3, 5)  # five native 336 crops per 512 input
    assert torch.all(prepared["image_sizes"] == 512)
    mask = prepared["attention_mask"]
    last = mask.sum(1) - 1
    assert torch.all(prepared["input_ids"][torch.arange(2), last] == processor.tokenizer.eos_token_id)
    assert prepared["input_ids"].shape[1] < c.max_length
    tiny_limit = c.model_copy(update={"max_length": 64})
    with pytest.raises(ValueError, match="do not truncate"):
        implementation.preprocess_llava(processor, tiny_limit, [[image]], ["red"], query=True)
    with pytest.raises(ValueError, match="placeholders"):
        implementation.preprocess_llava(processor, c, [[image]], ["<image>"], query=True)


def test_worker_preprocessing_matches_direct(monkeypatch):
    student, c = build_student(monkeypatch)
    images = [Image.new("RGB", (48, 32), (100, 20, 70)), Image.new("RGB", (32, 64), (20, 190, 40))]
    processor = student.embedder.processor
    direct = student.prepare_query_batch(images, ["red", "make it blue"])
    _, worker = QueryCollator(processor, c.model)([(0, images[0], "red"), (1, images[1], "make it blue")])
    assert direct.keys() == worker.keys()
    for key in direct:
        torch.testing.assert_close(direct[key], worker[key], rtol=0, atol=0)
    direct = student.prepare_image_batch(images)
    _, worker = ImageCollator(processor, c.model)(list(enumerate(images)))
    for key in direct:
        torch.testing.assert_close(direct[key], worker[key], rtol=0, atol=0)


def test_attention_restart_retains_initial_cache(tmp_path):
    from cir_qdiag_train.resume_config import restart_llava_attention
    current = config().model_dump(mode="json")
    previous = deepcopy(current)
    previous["model"].update(lora_target_modules="all-linear", lora_rank=32, lora_alpha=64)
    mining = tmp_path / "on_policy/epoch_001"
    mining.mkdir(parents=True)
    metadata = mining / "metadata.json"
    metadata.write_text(json.dumps({"complete": True, "source_global_step": 0}))
    effective = tmp_path / "effective_config.json"
    effective.write_text(json.dumps(previous))
    checkpoint = tmp_path / "last.pt"
    checkpoint.touch()
    with pytest.raises(RuntimeError, match="checkpoints"):
        restart_llava_attention(previous, current, tmp_path)
    assert effective.exists()
    checkpoint.unlink()
    metadata.write_text(json.dumps({"complete": True, "source_global_step": 10}))
    with pytest.raises(RuntimeError, match="global_step=0"):
        restart_llava_attention(previous, current, tmp_path)
    metadata.write_text(json.dumps({"complete": True, "source_global_step": 0}))
    assert restart_llava_attention(previous, current, tmp_path)
    assert metadata.exists() and not effective.exists()
    assert len(list(tmp_path.glob("previous_all_linear_*/effective_config.json"))) == 1


def test_attention_only_both_branches_gradients(monkeypatch):
    s, _ = build_student(monkeypatch, targets="attention")
    audit = s.trainability_audit()
    assert len(audit["modules"]["language"]) == 8
    assert not audit["modules"]["vision"] and not audit["modules"]["projector"]
    im = Image.new("RGB", (32, 32), (130, 90, 25))
    for encode, x in [(s.encode_prepared_queries, s.prepare_query_batch([im], ["red"])),
                      (s.encode_prepared_images, s.prepare_image_batch([im]))]:
        s.model.zero_grad(set_to_none=True)
        s.train()
        (encode(x, requires_grad=True) * torch.arange(32)).sum().backward()
        trainable = [(n, p) for n, p in s.model.named_parameters() if p.requires_grad]
        assert all("language_model" in n and ".self_attn." in n for n, _ in trainable)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for _, p in trainable)
        assert any(p.grad.abs().sum() > 0 for _, p in trainable)
        assert all(p.grad is None for p in s.model.parameters() if not p.requires_grad)


def test_both_branches_mlp_and_projector_gradients(monkeypatch):
    s, _ = build_student(monkeypatch)
    im = Image.new("RGB", (32, 32), (130, 90, 25))
    audit = s.trainability_audit()
    assert len(audit["inactive_vision_adapter_modules"]) == 6
    for encode, x in [(s.encode_prepared_queries, s.prepare_query_batch([im], ["red"])),
                      (s.encode_prepared_images, s.prepare_image_batch([im]))]:
        s.model.zero_grad(set_to_none=True)
        s.train()
        result = encode(x, requires_grad=True)
        assert torch.isfinite(result).all()
        (result * torch.arange(32)).sum().backward()
        for fragment in ("vision_tower", "language_model", "multi_modal_projector", ".gate_proj", ".up_proj", ".down_proj"):
            grads = [p.grad for n, p in s.model.named_parameters() if p.requires_grad and fragment in n]
            assert grads and any(g is not None and g.abs().sum() > 0 for g in grads)
        inactive = audit["inactive_vision_adapter_modules"]
        for n, p in s.model.named_parameters():
            if not p.requires_grad:
                assert p.grad is None
            elif not any(n.startswith(prefix + ".") for prefix in inactive):
                assert p.grad is not None and torch.isfinite(p.grad).all(), n


def test_padding_multi_image_and_shared_eval(monkeypatch):
    s, _ = build_student(monkeypatch)
    im = Image.new("RGB", (32, 32), (90, 190, 20))
    s.eval()
    singles = [s.encode_prepared_queries(s.prepare_query_batch([im], [text]), requires_grad=False)
               for text in ["red", "make the green background completely blue"]]
    batched = s.encode_prepared_queries(s.prepare_query_batch([im, im], ["red", "make the green background completely blue"]), requires_grad=False)
    torch.testing.assert_close(torch.cat(singles), batched, rtol=1e-4, atol=1e-5)
    multi = s.encode_prepared_queries(s.prepare_composed_query_batch([[im, im]], ["combine"]), requires_grad=False)
    assert multi.shape == (1, 32) and torch.isfinite(multi).all()
    prepared = s.prepare_image_batch([im])
    with torch.no_grad():
        for n, p in s.model.named_parameters():
            if "lora_B" in n:
                p.add_(torch.randn_like(p) * .1)
    enabled = s.encode_prepared_images(prepared)
    with s.model.disable_adapter():
        disabled = s.encode_prepared_images(prepared)
    assert not torch.allclose(enabled, disabled)


def test_checkpoint_and_optimizer_resume(monkeypatch, tmp_path):
    s, c = build_student(monkeypatch)
    c.experiment.output_dir = tmp_path
    im = Image.new("RGB", (32, 32), (90, 20, 190))
    opt = torch.optim.AdamW(s.trainable_parameters(), lr=5e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
    x = s.prepare_image_batch([im])
    (s.encode_prepared_images(x, requires_grad=True) * torch.arange(32)).sum().backward()
    opt.step()
    scheduler.step()
    path = tmp_path / "last.pt"
    save_qwen_checkpoint(path, student=s, optimizer=opt, scheduler=scheduler,
                         epoch=0, global_step=1, config=c.model_dump(mode="json"))
    s.eval()
    expected = s.encode_prepared_images(x)
    restored, _ = qwen_student_from_checkpoint(path, device=torch.device("cpu"), mixed_precision="no")
    torch.testing.assert_close(expected, restored.encode_prepared_images(x))
    new_opt = torch.optim.AdamW(restored.trainable_parameters(), lr=1.)
    new_sched = torch.optim.lr_scheduler.StepLR(new_opt, step_size=1)
    load_qwen_checkpoint(path, student=restored, optimizer=new_opt, scheduler=new_sched,
                         expected_config=c.model_dump(mode="json"))
    assert new_opt.param_groups[0]["lr"] == opt.param_groups[0]["lr"]
    assert new_sched.state_dict() == scheduler.state_dict()


@pytest.mark.parametrize("legacy_keys", [False, True])
def test_native_weight_loading(tmp_path, legacy_keys):
    from safetensors.torch import save_file

    reference = LlavaNextForConditionalGeneration(tiny_config())
    state = reference.state_dict()
    if legacy_keys:
        state = {k.replace("model.language_model.", "language_model.model.")
                 .replace("model.vision_tower.", "vision_tower.")
                 .replace("model.multi_modal_projector.", "multi_modal_projector.")
                 .replace("model.image_newline", "image_newline")
                 .replace("lm_head.weight", "language_model.lm_head.weight"): v for k, v in state.items()}
    # save_pretrained itself converts native names back to legacy names. Write
    # these explicit fixtures directly so legacy keys are not converted twice.
    reference.config.save_pretrained(tmp_path)
    save_file(state, tmp_path / "model.safetensors", metadata={"format": "pt"})
    c = config().model.model_copy(update={"pretrained_model_name_or_path": str(tmp_path)})
    loaded = implementation.load_llava_backbone(c, torch.float32)
    for name, p in reference.model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], p, rtol=0, atol=0)


def test_shared_collator_deduplicates(tmp_path):
    c = config().model
    processor = implementation.load_llava_processor(c)
    paths = [tmp_path / f"{i}.png" for i in range(3)]
    for i, p in enumerate(paths):
        Image.new("RGB", (32, 32), (i * 50, 10, 20)).save(p)
    indices, prepared = SharedTrainingCollator(processor, c, image_batch_size=2)([
        (10, paths[0], "red", [paths[1], paths[2]]),
        (20, paths[1], "blue", [paths[2], paths[0]]),
    ])
    assert indices == [10, 20]
    assert prepared["candidate_positions"].tolist() == [[0, 1], [1, 2]]
    assert [len(b["image_sizes"]) for b in prepared["images"]] == [2, 1]


def test_dry_run():
    r = subprocess.run([sys.executable, str(PROJECT / "scripts/run_llava_lora_e4.py"), "--dry-run"],
                       cwd=PROJECT, capture_output=True, text=True, check=True, timeout=30)
    assert r.stdout.count("batch=16 x 1; GPUs=2; global_batch=32; K=32; candidate_chunk=8; gradient_cache=True; workers_per_rank=8; prefetch=1") == 5


@pytest.mark.parametrize("suffix", SUFFIXES)
def test_adapter_routing(monkeypatch, suffix):
    monkeypatch.syspath_prepend(str(PROJECT.parents[1] / "CIR_PLAY"))
    from run_pinpoint_bge_lincir import infer_adapter_kind
    path = PROJECT / "tests/test_llava_lora.py"  # existing file; don't load it as weights
    assert infer_adapter_kind(f"LLAVA-llava_{suffix}", path) == "trained_llava_embedding"


@pytest.mark.parametrize("suffix", ["oprd_s2", "infonce", "cmse_s2"])
def test_tiny_native_training_loop_all_objectives(monkeypatch, tmp_path, suffix):
    from cir_qdiag_train import qwen_on_policy_trainer as trainer

    student, c = build_student(monkeypatch)
    c.objective = config(suffix).objective
    c.mining.candidate_source = config(suffix).mining.candidate_source
    c.experiment.output_dir = tmp_path / "run"
    c.data.num_workers = 0
    c.optimizer.epochs = 1
    c.optimizer.mixed_precision = "no"
    paths = [tmp_path / f"{i}.png" for i in range(40)]
    for i, path in enumerate(paths):
        Image.new("RGB", (32, 32), (i * 5, 200 - i * 4, i * 3)).save(path)
    records = [SimpleNamespace(reference_id="0", target_id=str(i), reference_path=paths[0], text="edit")
               for i in [1, 2, 3, 4]]
    gallery_refreshes = []
    original_encode = trainer.encode_current_gallery

    def encode_gallery(*args):
        gallery_refreshes.append(args[-1])
        return original_encode(*args)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(implementation, "LlavaLoRAStudent", lambda *a, **kw: student)
    monkeypatch.setattr(trainer, "load_image_map", lambda *a: ([str(i) for i in range(40)], paths))
    monkeypatch.setattr(trainer, "load_cirr_positive_records", lambda *a: records)
    monkeypatch.setattr(trainer, "encode_current_gallery", encode_gallery)
    monkeypatch.setattr(trainer, "_load_student_gallery", lambda *a: pytest.fail("fixed gallery used"))
    teacher_scores = np.random.default_rng(42).normal(size=(4, 32)).astype(np.float32)
    monkeypatch.setattr(trainer, "_score_teacher_candidates", lambda **kw: (teacher_scores, np.ones(4, dtype=bool)))
    checkpoint = trainer.train_qwen_on_policy_distillation(c)
    payload = torch.load(checkpoint, weights_only=False, map_location="cpu")
    assert payload["global_step"] == 1 and payload["epoch"] == 0
    assert gallery_refreshes == ([1] if suffix == "oprd_s2" else [])
    assert any(p.count_nonzero() for n, p in student.model.named_parameters() if "lora_B" in n)
    for fragment in ("language_model", "vision_tower", "multi_modal_projector"):
        assert any(p.count_nonzero() for n, p in student.model.named_parameters() if "lora_B" in n and fragment in n)
    metadata = json.loads((c.experiment.output_dir / "on_policy/epoch_001/metadata.json").read_text())
    assert metadata["student_variant"] == "llava_shared_lora"


def test_downloaded_weight_shapes_and_real_adapter_count(monkeypatch):
    from safetensors import safe_open
    c = config().model
    root = Path(c.pretrained_model_name_or_path)
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    with torch.device("meta"):
        native = LlavaNextModel(LlavaNextConfig.from_pretrained(root, local_files_only=True))
    shapes = {}
    for shard in set(index.values()):
        with safe_open(root / shard, framework="pt", device="cpu") as f:
            for name in f.keys():
                if name == "language_model.lm_head.weight":
                    continue
                mapped = name.replace("language_model.model.", "language_model.")
                shapes[mapped] = tuple(f.get_slice(name).get_shape())
    expected = {n: tuple(p.shape) for n, p in native.state_dict().items()}
    assert shapes == expected
    monkeypatch.setattr(implementation, "load_llava_backbone", lambda *args: native)
    with torch.device("meta"):
        student = implementation.LlavaLoRAStudent(c, device=torch.device("meta"), mixed_precision="no")
    audit = student.trainability_audit()
    assert {k: len(v) for k, v in audit["modules"].items()} == {"language": 224, "vision": 144, "projector": 2}
    assert audit["trainable_parameter_count"] == 98467840


def test_runner_resumes_and_evaluates_four_tasks(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("llava_test_runner", PROJECT / "scripts/run_main_comparison_e4.py")
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)
    c = config()
    c.experiment.output_dir = tmp_path
    name, label = "llava_oprd_s2", "LLAVA-llava_oprd_s2"
    monkeypatch.setattr(suite, "load", lambda _: c)
    monkeypatch.setattr(suite.runner, "LABEL_PREFIX", "LLAVA")
    monkeypatch.setitem(suite.runner.CONFIG_PATHS, name, PROJECT / f"configs/{name}.yaml")
    (tmp_path / "last.pt").touch()
    commands = []
    def execute(command, *args, **kwargs):
        command = list(map(str, command))
        commands.append(command)
        if any("train_llava" in value for value in command):
            assert command[1:3] == ["-m", "torch.distributed.run"]
            assert "--nproc_per_node=2" in command
            assert "--resume-from" in command
            (tmp_path / "checkpoint_epoch_002.pt").touch()
            return
        assert command[command.index("--model-path") + 1].endswith("checkpoint_epoch_002.pt") if "--model-path" in command else "checkpoint_epoch_002.pt" in command[command.index("--adapter") + 1]
        target = (Path(command[command.index("--output-dir") + 1]) if "--output-dir" in command
                  else Path(command[command.index("--output-root") + 1]) / label)
        target.mkdir(parents=True, exist_ok=True)
        filename = ("val_metrics.json" if "evaluate_cirr" in command[1] else "metrics.json" if "fashioniq" in command[1]
                    else "official_metrics.csv" if "pinpoint" in command[1] else "summary.csv")
        (target / filename).write_text("{}")
    monkeypatch.setattr(suite.runner, "execute", execute)
    suite.run_qwen(name)
    assert len(commands) == 5
    suite.run_qwen(name)
    assert len(commands) == 5
