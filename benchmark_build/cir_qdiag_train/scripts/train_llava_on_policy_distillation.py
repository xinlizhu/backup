"""LLaVA entry point for the shared multimodal distillation engine."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cir_qdiag_train.qwen_on_policy_config import load_qwen_on_policy_config
from cir_qdiag_train.qwen_on_policy_trainer import train_qwen_on_policy_distillation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    config = load_qwen_on_policy_config(args.config)
    if not config.model.llava:
        parser.error("requires a llava_shared_lora config")
    if config.optimizer.world_size > 1:
        from cir_qdiag_train.llava_parallel_trainer import train_llava_parallel
        checkpoint = train_llava_parallel(config, resume_from=args.resume_from, smoke_only=args.smoke_only)
    else:
        if args.smoke_only:
            parser.error("parallel smoke requires world_size > 1")
        checkpoint = train_qwen_on_policy_distillation(config, resume_from=args.resume_from)
    if checkpoint is not None:
        print(checkpoint)


if __name__ == "__main__":
    main()
