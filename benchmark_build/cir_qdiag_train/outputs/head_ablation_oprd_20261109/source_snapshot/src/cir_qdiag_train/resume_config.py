"""Loader concurrency may change on epoch resume; training semantics may not."""

from copy import deepcopy


def without_loader_settings(payload):
    result = deepcopy(payload)
    for key in ("num_workers", "prefetch_factor"):
        result.get("data", {}).pop(key, None)
    return result


def restart_llava_attention(previous, current, output):
    """Caller holds the run lock. Reuse only initial mining, never old adapters."""
    import json
    import shutil
    import tempfile
    from pathlib import Path

    old, new = without_loader_settings(previous), without_loader_settings(current)
    if not (old["model"]["variant"] == new["model"]["variant"] == "llava_shared_lora"
            and old["model"]["lora_target_modules"] == "all-linear"
            and new["model"]["lora_target_modules"] == "attention"
            and new["model"]["lora_rank"] == 16):
        return False
    for key in ("lora_target_modules", "lora_rank", "lora_alpha"):
        old["model"][key] = new["model"][key]
    if old != new:
        return False
    output = Path(output)
    if list(output.glob("*.pt")):
        raise RuntimeError("Cannot reuse all-linear checkpoints with attention rank16; archive checkpoints before restarting")
    mining = output / "on_policy"
    if mining.exists():
        for entry in mining.iterdir():
            if entry.name != "epoch_001":
                raise RuntimeError("Only initial epoch_001 mining can be reused after changing adapters")
            metadata = entry / "metadata.json"
            if not metadata.is_file():
                raise RuntimeError("Initial mining metadata missing; cannot verify cache provenance")
            info = json.loads(metadata.read_text())
            if not info.get("complete") or info.get("source_global_step") != 0:
                raise RuntimeError("Only complete pre-training mining at global_step=0 can be reused")
    archive = Path(tempfile.mkdtemp(prefix="previous_all_linear_", dir=output))
    for name in ("effective_config.json", "runner.log", "adapter_audit.json",
                 "gradient_audit.json", "run_summary.json", "train_metrics.jsonl"):
        source = output / name
        if source.is_file():
            shutil.move(str(source), str(archive / name))
    print(f"[restart attention] old records archived to {archive}; initial mining retained", flush=True)
    return True
