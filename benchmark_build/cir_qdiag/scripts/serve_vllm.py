from __future__ import annotations

import argparse
import json
import os
import shlex
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Start Qwen3.6-27B with vLLM for CIR-QDiag")
    parser.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--tensor-parallel-size", type=int, default=0)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--mm-processor-max-pixels", type=int)
    parser.add_argument(
        "--gdn-prefill-backend",
        choices=("triton", "flashinfer"),
        help="Override the GDN prefill backend configured in model.yaml.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    mm_processor_kwargs = dict(cfg.get("mm_processor_kwargs", {}))
    if args.mm_processor_max_pixels is not None:
        mm_processor_kwargs["max_pixels"] = args.mm_processor_max_pixels
    tp = args.tensor_parallel_size or visible_gpu_count()
    if tp < 1:
        raise SystemExit("No visible GPU. Set CUDA_VISIBLE_DEVICES or --tensor-parallel-size.")
    max_model_len = args.max_model_len or cfg.get("max_model_len", 16384)
    max_num_batched_tokens = args.max_num_batched_tokens or cfg.get(
        "max_num_batched_tokens", max_model_len
    )
    command = [
        "vllm",
        "serve",
        str(cfg["model_path"]),
        "--served-model-name",
        str(cfg["served_model_name"]),
        "--host",
        str(cfg.get("host", "127.0.0.1")),
        "--port",
        str(cfg.get("port", 8000)),
        "--tensor-parallel-size",
        str(tp),
        "--dtype",
        str(cfg.get("dtype", "bfloat16")),
        "--max-model-len",
        str(max_model_len),
        "--max-num-batched-tokens",
        str(max_num_batched_tokens),
        "--max-num-seqs",
        str(args.max_num_seqs or cfg.get("max_num_seqs", 8)),
        "--gpu-memory-utilization",
        str(cfg.get("gpu_memory_utilization", 0.9)),
        "--generation-config",
        "vllm",
        "--limit-mm-per-prompt",
        '{"image": 2}',
        "--allowed-local-media-path",
        str(cfg["image_root"]),
        "--reasoning-parser",
        "qwen3",
        "--gdn-prefill-backend",
        str(args.gdn_prefill_backend or cfg.get("gdn_prefill_backend", "triton")),
    ]
    if mm_processor_kwargs:
        command.extend(
            [
                "--mm-processor-kwargs",
                json.dumps(mm_processor_kwargs, sort_keys=True, separators=(",", ":")),
            ]
        )
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        os.execvp(command[0], command)


def visible_gpu_count() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        values = [
            item.strip() for item in visible.split(",") if item.strip() and item.strip() != "-1"
        ]
        return len(values)
    try:
        import torch

        return torch.cuda.device_count()
    except ImportError:
        return 0


if __name__ == "__main__":
    main()
