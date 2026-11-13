from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from cir_qdiag.data.cirr_loader import CIRRDataError, load_cirr_annotations


def _dataset(tmp_path: Path, target_field: str = "target_hard", duplicate: bool = False):
    root = tmp_path / "images"
    root.mkdir()
    for name, color in (("ref.png", "red"), ("target.png", "blue")):
        Image.new("RGB", (8, 8), color).save(root / name)
    mapping = {"ref": "./ref.png", "target": "./target.png"}
    row = {
        "pairid": 7,
        "reference": "ref",
        target_field: "target",
        "caption": "make it blue",
        "img_set": {"members": ["ref", "target"]},
    }
    annotations = [row, dict(row)] if duplicate else [row]
    cap = tmp_path / "cap.json"
    split = tmp_path / "split.json"
    cap.write_text(json.dumps(annotations), encoding="utf-8")
    split.write_text(json.dumps(mapping), encoding="utf-8")
    return cap, split, root


@pytest.mark.parametrize("target_field", ["target_hard", "target"])
def test_target_aliases(tmp_path: Path, target_field: str):
    cap, split, root = _dataset(tmp_path, target_field)
    records = load_cirr_annotations(cap, root, "val", split)
    assert len(records) == 1
    assert records[0].target.image_id == "target"


def test_duplicate_pairid_is_rejected(tmp_path: Path):
    cap, split, root = _dataset(tmp_path, duplicate=True)
    with pytest.raises(CIRRDataError, match="duplicate pairid"):
        load_cirr_annotations(cap, root, "val", split)


def test_train_split_uses_train_identity(tmp_path: Path):
    cap, split, root = _dataset(tmp_path)
    record = load_cirr_annotations(cap, root, "train", split)[0]
    assert record.split == "train"
    assert record.sample_id == "cirr_train_7"


def test_missing_image_is_rejected(tmp_path: Path):
    cap, split, root = _dataset(tmp_path)
    (root / "target.png").unlink()
    with pytest.raises(CIRRDataError, match="missing image"):
        load_cirr_annotations(cap, root, "val", split)


def test_test_split_is_forbidden(tmp_path: Path):
    cap, split, root = _dataset(tmp_path)
    with pytest.raises(CIRRDataError, match="validation split"):
        load_cirr_annotations(cap, root, "test1", split)


def test_non_string_caption_is_rejected(tmp_path: Path):
    cap, split, root = _dataset(tmp_path)
    rows = json.loads(cap.read_text(encoding="utf-8"))
    rows[0]["caption"] = None
    cap.write_text(json.dumps(rows), encoding="utf-8")

    with pytest.raises(CIRRDataError, match="empty caption"):
        load_cirr_annotations(cap, root, "val", split)


@pytest.mark.parametrize("unsafe", ["/etc/passwd", "../target.png"])
def test_image_mapping_cannot_escape_image_root(tmp_path: Path, unsafe: str):
    cap, split, root = _dataset(tmp_path)
    mapping = json.loads(split.read_text(encoding="utf-8"))
    mapping["target"] = unsafe
    split.write_text(json.dumps(mapping), encoding="utf-8")

    with pytest.raises(CIRRDataError, match="unsafe image path"):
        load_cirr_annotations(cap, root, "val", split)
