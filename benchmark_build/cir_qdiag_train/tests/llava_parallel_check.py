"""CPU-only torchrun integration checks; never load the 7B weights or use CUDA."""
import argparse
import json
import os
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import numpy as np
import pytest
import torch
import torch.distributed as dist
from PIL import Image

from cir_qdiag_train.llava_parallel import (
    average_metrics, broadcast_parameters, on_rank_zero, setup_parallel, shard_order, sum_gradients,
    OverlappedGradientSum,
)
from cir_qdiag_train.qwen_gradient_cache import shared_student_scores


class ToyStudent:
    def __init__(self):
        self.model = torch.nn.Linear(3, 4)
        self.model.register_parameter("unused", torch.nn.Parameter(torch.ones(2)))
        self.device = torch.device("cpu")

    def encode_prepared_queries(self, prepared, requires_grad=True):
        return torch.nn.functional.normalize(self.model(prepared), dim=-1)

    encode_prepared_images = encode_prepared_queries


def gradients():
    rank, world, device = setup_parallel(2, allow_cpu=True)
    try:
        for replay in (False, True, "overlap"):
            for objective in ("kl", "infonce", "cmse"):
                torch.manual_seed(91 + rank)
                student = ToyStudent()
                broadcast_parameters(student.model.parameters())
                reference = deepcopy(student)
                reducer = OverlappedGradientSum(student.model.parameters(), bucket_bytes=32) if replay == "overlap" else None
                generator = torch.Generator().manual_seed(52)
                queries = torch.randn(35, 3, generator=generator)
                images = torch.randn(40, 3, generator=generator)
                candidates = torch.randint(40, (35, 32), generator=generator)
                teacher = torch.randn(35, 32, generator=generator)

                def score(s, indices, cache=False):
                    return shared_student_scores(s, {"queries": queries[indices],
                                                 "images": [images[:20], images[20:]],
                                                 "candidate_positions": candidates[indices]}, gradient_cache=cache)

                def loss(scores, indices):
                    if objective == "infonce":
                        return torch.nn.functional.cross_entropy(scores / .07, torch.zeros(len(indices), dtype=torch.long))
                    if objective == "cmse":
                        a, b = scores - scores.mean(-1, keepdim=True), teacher[indices] - teacher[indices].mean(-1, keepdim=True)
                        return (a - b).square().mean()
                    logp, logq = (scores / .05).log_softmax(-1), (teacher[indices] / .05).log_softmax(-1)
                    return (logp.exp() * (logp - logq)).sum(-1).mean()

                # Global 32 followed by only three queries: rank 1 has no data.
                for start in (0, 32):
                    indices = list(range(start, min(start + 32, 35)))
                    local = shard_order(indices, 16, rank, world)
                    student.model.zero_grad(set_to_none=True)
                    reference.model.zero_grad(set_to_none=True)
                    if reducer:
                        reducer.begin()
                    serial, _ = score(reference, indices)
                    loss(serial, indices).backward()
                    if local:
                        values, backward = score(student, local, replay is True)
                        (loss(values, local) * len(local) / len(indices)).backward()
                        backward()
                    if reducer:
                        # The deliberately unused parameter can stall the first
                        # ordered bucket; finish must still preserve grad=None.
                        reducer.finish()
                    else:
                        sum_gradients(student.model.parameters(), bucket_bytes=32)
                    for p, expected in zip(student.model.parameters(), reference.model.parameters(), strict=True):
                        if expected.grad is None:
                            assert p.grad is None
                        else:
                            torch.testing.assert_close(p.grad, expected.grad, atol=2e-5, rtol=1e-5)
                mean = average_metrics({"loss": 2.} if rank == 0 else {}, 3 if rank == 0 else 0, device)
                assert mean == {"loss": 2.}
                if reducer:
                    reducer.close()
        # Real Qwen shared multi-forward graph with non-reentrant checkpointing.
        from test_qwen_shared_lora import build_tiny_student, tiny_inputs
        with pytest.MonkeyPatch.context() as patch:
            torch.manual_seed(17)
            student = build_tiny_student(patch)
            student.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            parameters = student.trainable_parameters()
            broadcast_parameters(parameters)
            reducer = OverlappedGradientSum(parameters, bucket_bytes=1024)
            for iteration in range(2):
                inputs = tiny_inputs()
                student.model.zero_grad(set_to_none=True)
                rng = torch.get_rng_state()
                def objective():
                    q = student.encode_prepared_queries(inputs, requires_grad=True)
                    c1 = student.encode_prepared_images(inputs, requires_grad=True)
                    c2 = student.encode_prepared_images(inputs, requires_grad=True)
                    return ((q * c1).sum() + (q * c2).sum()) * (rank + 1) / world
                objective().backward()
                sum_gradients(parameters)
                expected = [p.grad.clone() if p.grad is not None else None for p in parameters]
                student.model.zero_grad(set_to_none=True)
                torch.set_rng_state(rng)
                reducer.begin()
                objective().backward()
                assert reducer.pending
                reducer.finish()
                for p, wanted in zip(parameters, expected, strict=True):
                    if wanted is None:
                        assert p.grad is None
                    else:
                        torch.testing.assert_close(p.grad, wanted, atol=2e-5, rtol=1e-5)
            reducer.close()
        with pytest.raises(RuntimeError, match="intentional"):
            on_rank_zero(lambda: (_ for _ in ()).throw(ValueError("intentional")))
        dist.barrier()
        if rank == 0:
            print("GRADIENT_EQUIVALENCE_OK", flush=True)
    finally:
        dist.destroy_process_group()


def training(root, suffix, phase):
    from test_llava_lora import build_student, config
    from cir_qdiag_train import llava_parallel_trainer as trainer

    rank = int(os.environ["RANK"])
    # Production resume starts a fresh torchrun job. Do the same here; Gloo
    # cannot safely reuse a destroyed group's rendezvous keys in this job.
    for suffix in (suffix,):
        with pytest.MonkeyPatch.context() as patch:
            torch.manual_seed(91)
            student, c = build_student(patch)
            c.objective = config(suffix).objective
            c.mining.candidate_source = config(suffix).mining.candidate_source
            c.optimizer.world_size = 2
            c.optimizer.batch_size_per_gpu = 2
            c.optimizer.epochs = 2
            c.optimizer.save_every_epochs = 1
            c.optimizer.mixed_precision = "no"
            c.data.num_workers = 0
            c.experiment.output_dir = root / suffix
            paths = [root / f"{i}.png" for i in range(40)]
            if rank == 0:
                root.mkdir(parents=True, exist_ok=True)
                for i, path in enumerate(paths):
                    Image.new("RGB", (32, 32), (i * 5, 200 - i * 4, i * 3)).save(path)
            records = [SimpleNamespace(reference_id="0", target_id=str(i), reference_path=paths[0], text="edit")
                       for i in range(1, 6)]
            holder = [student]
            patch.setattr(trainer, "LlavaLoRAStudent", lambda *a, **kw: holder[0])
            patch.setattr(trainer.engine, "load_image_map", lambda *a: ([str(i) for i in range(40)], paths))
            patch.setattr(trainer.engine, "load_cirr_positive_records", lambda *a: records)
            patch.setattr(trainer.engine, "_score_teacher_candidates", lambda **kw:
                          (np.random.default_rng(42).normal(size=(5, 32)).astype(np.float32), np.ones(5, dtype=bool)))
            save_original = trainer.save_qwen_checkpoint
            def save(path, **kwargs):
                assert rank == 0
                save_original(path, **kwargs)
                if path.name == "last.pt" and kwargs["epoch"] == 0:
                    shutil.copy2(path, path.with_name("resume_epoch001.pt"))
            patch.setattr(trainer, "save_qwen_checkpoint", save)
            checkpoint = trainer.train_llava_parallel(
                c, allow_cpu=True,
                resume_from=c.experiment.output_dir / "resume_epoch001.pt" if phase == "resume" else None)
            actual = student.lora_state_dict()
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            assert payload["epoch"] == 1 and payload["global_step"] == 4
            for key, value in actual.items():
                torch.testing.assert_close(value, payload["model"][key], rtol=0, atol=0)
            expected_path = c.experiment.output_dir / f"expected_rank{rank}.pt"
            if phase == "full":
                torch.save(actual, expected_path)
            else:
                expected = torch.load(expected_path, map_location="cpu", weights_only=True)
                for key, value in expected.items():
                    torch.testing.assert_close(value, actual[key], rtol=0, atol=0)
            summary = json.loads((c.experiment.output_dir / "run_summary.json").read_text())
            assert summary["effective_batch_size"] == 4 and summary["optimizer_steps_per_epoch"] == 2
    if rank == 0:
        print("TINY_TRAINING_AND_RESUME_OK", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["gradients", "training"], required=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--suffix", default="oprd_s2")
    parser.add_argument("--phase", choices=["full", "resume"], default="full")
    args = parser.parse_args()
    assert not torch.cuda.is_available(), "This integration check must be CPU-only"
    gradients() if args.mode == "gradients" else training(args.work_dir, args.suffix, args.phase)
