"""Launch one synchronized update on the configured LLaVA GPUs, saving no checkpoint."""
import argparse
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from cir_qdiag_train.qwen_on_policy_config import load_qwen_on_policy_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/llava_oprd_s2.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_qwen_on_policy_config(args.config)
    if not config.model.llava or config.optimizer.world_size < 2:
        parser.error("requires a multi-GPU LLaVA config")
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               f"--nproc_per_node={config.optimizer.world_size}", "--max_restarts=0",
               str(PROJECT / "scripts/train_llava_on_policy_distillation.py"),
               "--config", str(args.config.resolve()), "--smoke-only"]
    print(f"Smoke: {config.optimizer.world_size} GPUs x {config.optimizer.batch_size_per_gpu} queries, "
          f"accumulation=1; global_batch={config.optimizer.world_size * config.optimizer.batch_size_per_gpu}; "
          f"candidate_chunk={config.optimizer.candidate_encode_batch_size}; "
          f"gradient_cache={config.optimizer.candidate_gradient_cache}", flush=True)
    if args.dry_run:
        print(command)
    else:
        subprocess.run(command, check=True, cwd=PROJECT)


if __name__ == "__main__":
    main()
