#!/usr/bin/env python3
"""Evaluate a locally trained TG-CIR checkpoint on CIRR/FashionIQ."""

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import torch

from cir_qdiag_train.tg_cir_data import TGCIRProcessor
from cir_qdiag_train.tg_cir_evaluation import evaluate_cirr, evaluate_fashioniq
from cir_qdiag_train.tg_cir_runtime import build_model, evaluation_identity, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", choices=["cirr", "fashioniq"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int)
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU visible")
    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint["config"]
    if args.num_workers is not None:
        if args.num_workers < 0:
            parser.error("num-workers must be >= 0")
        config["evaluation"]["num_workers"] = args.num_workers
    evaluation_identity(args.output_dir, args.checkpoint, config, args.dataset)
    result_path = args.output_dir / "metrics.json"
    if result_path.is_file():
        print(result_path.read_text())
        return
    model = build_model(config, checkpoint["clip_config"])
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model.to(device).eval()
    processor = TGCIRProcessor(config["model"]["pretrained_model_name_or_path"])
    if args.dataset == "cirr":
        metrics = evaluate_cirr(model, processor, config, device, args.output_dir)
    else:
        metrics = evaluate_fashioniq(
            model, processor, config, device, args.output_dir, args.checkpoint
        )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
