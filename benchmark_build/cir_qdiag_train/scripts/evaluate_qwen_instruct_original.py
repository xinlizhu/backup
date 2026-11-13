"""Evaluate original Qwen3-VL-2B-Instruct using exactly the student's preprocessing and pooling."""
import argparse
import gc
import fcntl
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))
os.environ["PYTHONPATH"] = str(PROJECT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
import torch
from cir_qdiag_train.qwen_student import QwenQueryLoRAStudent
from cir_qdiag_train.qwen_on_policy_config import load_qwen_on_policy_config
from cir_qdiag_train.qwen_student import load_qwen_checkpoint, save_qwen_checkpoint
from cir_qdiag_train.qwen_on_policy_trainer import _set_seed
from cir_qdiag_train.on_policy_mining import write_json_atomic
from run_main_comparison_e4 import runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/qwen2_instruct_shared_oprd_s2.yaml")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "outputs/qwen_instruct_original")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--only-cirr", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    c = load_qwen_on_policy_config(args.config)
    if c.model.variant != "qwen3_vl_shared_lora" or not c.model.pretrained_model_name_or_path.endswith("Qwen3-VL-2B-Instruct"):
        parser.error("requires a Qwen3-VL-2B-Instruct config")
    args.output_dir = args.output_dir.resolve()
    args.config = args.config.resolve()
    checkpoint = args.output_dir / "initialization.pt"
    label = "Qwen3-VL-2B-Instruct-original"
    batch = str(c.evaluation.batch_size)
    cirr, fashion = args.output_dir / "cirr_val", args.output_dir / "fashioniq"
    commands = [
        (cirr / "val_metrics.json", [sys.executable, PROJECT / "scripts/evaluate_cirr_original_adapter.py",
         "--config", args.config, "--model-name", label, "--model-path", checkpoint,
         "--output-dir", cirr, "--device", args.device, "--dtype", "auto",
         "--image-batch-size", batch, "--query-batch-size", batch]),
    ]
    if not args.only_cirr:
        diag = PROJECT.parent / "diag_model"
        commands.extend([
            (fashion / "metrics.json", [sys.executable, PROJECT / "scripts/evaluate_fashioniq_original_adapter.py",
             "--model-name", label, "--adapter-name", label, "--model-path", checkpoint,
             "--dataset-root", Path(os.environ.get("FASHIONIQ_ROOT", str(PROJECT.parents[1] / "CIR_new/datasets/fashion-iq"))), "--output-dir", fashion,
             "--device", args.device, "--image-batch-size", batch, "--query-batch-size", batch]),
            (args.output_dir / "pinpoint_official" / label / "official_metrics.csv", [sys.executable,
             diag / "run_clip_pinpoint_official.py", "--models", label, "--adapter", f"{label}={checkpoint}",
             "--output-root", args.output_dir / "pinpoint_official", "--device", args.device, "--batch-size", batch]),
            (args.output_dir / "qdiag" / label / "summary.csv", [sys.executable,
             diag / "evaluate_winrate_random10.py", "--adapter", f"{label}={checkpoint}",
             "--output-root", args.output_dir / "qdiag", "--device", args.device, "--batch-size", batch,
             "--random-samples", "10", "--seed", "42"]),
        ])
    if args.dry_run:
        for target, command in commands:
            print("[evaluate]", " ".join(map(str, command)), flush=True)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / ".baseline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        evaluate(args, c, checkpoint, commands)


def evaluate(args, c, checkpoint, commands):
    _set_seed(c.experiment.seed)
    student = QwenQueryLoRAStudent(c.model, device=torch.device(args.device), mixed_precision=c.optimizer.mixed_precision)
    if checkpoint.exists():
        payload = load_qwen_checkpoint(checkpoint, student=student)
        if payload.get("config", {}).get("model") != c.model.model_dump(mode="json"):
            raise ValueError("Baseline checkpoint model/preprocessing differs from current config")
        if payload["epoch"] != -1 or payload["global_step"] != 0:
            raise ValueError("Baseline checkpoint is not an initialization")
    student.eval()
    # Zero-initialized B makes the adapter an exact no-op, not a trained checkpoint.
    updates = [p for n, p in student.model.named_parameters() if "lora_B" in n]
    if not updates or any(bool(p.detach().count_nonzero()) for p in updates):
        raise ValueError("Baseline must have zero LoRA updates")
    if not checkpoint.exists():
        save_qwen_checkpoint(checkpoint, student=student, optimizer=None, scheduler=None,
                             epoch=-1, global_step=0, config=c.model_dump(mode="json"))
    write_json_atomic(args.output_dir / "baseline_metadata.json", {
        "initialization_only": True, "trained": False, "epoch": -1, "global_step": 0, "base_model": c.model.pretrained_model_name_or_path,
        "checkpoint": str(checkpoint), "preprocessing": c.model.model_dump(mode="json"),
    })
    del updates, student
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    with (args.output_dir / "runner.log").open("a", buffering=1) as log:
        for target, command in commands:
            if not target.is_file():
                runner.execute(command, log)


if __name__ == "__main__":
    main()
