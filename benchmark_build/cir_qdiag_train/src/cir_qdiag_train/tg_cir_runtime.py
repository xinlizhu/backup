"""TG-CIR construction, safe local checkpoint bookkeeping, and RNG restoration."""

import json
import os
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPConfig, CLIPModel

from cir_qdiag_train.tg_cir_config import clip_backbone_name, training_identity
from cir_qdiag_train.tg_cir_model import TGCIR


def build_model(config, clip_config=None):
    options = config["model"]
    if clip_config is None:
        clip = CLIPModel.from_pretrained(
            options["pretrained_model_name_or_path"],
            local_files_only=True,
            attn_implementation="eager",
        )
    else:
        backbone_config = CLIPConfig.from_dict(clip_config)
        backbone_config._attn_implementation = "eager"
        clip = CLIPModel(backbone_config)
    # L/14 is a backbone extension; attribute projections and heads derive their
    # dimensions from CLIP's config (B: 512; L: 768), not from fixed B constants.
    clip_backbone_name(clip.config.to_dict())
    if options["gradient_checkpointing"]:
        clip.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return TGCIR(
        clip.float(),
        options["global_tokens"],
        options["local_tokens"],
        options["temperature"],
    )


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def link_checkpoint(source, destination):
    """Immutable epoch/best aliases; future last.pt saves atomically replace its inode."""
    source, destination = Path(source), Path(destination)
    temporary = destination.with_name(destination.name + ".link.tmp")
    temporary.unlink(missing_ok=True)
    os.link(source, temporary)
    os.replace(temporary, destination)


def load_checkpoint(path, expected_config=None):
    # Only our own checkpoints should be supplied; never load third-party pickle weights.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("method") != "tg_cir_v1":
        raise ValueError("Not a locally trained TG-CIR checkpoint")
    if expected_config is not None and training_identity(payload["config"]) != training_identity(
        expected_config
    ):
        raise ValueError("Resume configuration changed; use a new output directory")
    return payload


@contextmanager
def output_lock(directory):
    import fcntl

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".train.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another TG-CIR trainer is using {directory}") from error
        yield


def evaluation_identity(directory, checkpoint, config, dataset="cirr"):
    """Reject stale embeddings after a checkpoint/config change, including last.pt."""
    from cir_qdiag_train.on_policy_mining import write_json_atomic

    checkpoint, directory = Path(checkpoint).resolve(), Path(directory)
    stat = checkpoint.stat()
    expected = {
        "checkpoint": str(checkpoint),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "config": training_identity(config),
        "dataset": dataset,
        "fashioniq_root": config["evaluation"]["fashioniq_root"],
        "protocol": "tg-cir-clip-zero-pad77-v1",
    }
    path = directory / "identity.json"
    if path.is_file() and json.loads(path.read_text()) != expected:
        raise ValueError(f"Evaluation inputs changed; choose a new output directory: {directory}")
    if not path.exists() and directory.exists() and any(directory.iterdir()):
        raise ValueError(f"Refusing unidentified existing evaluation cache: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, expected)
