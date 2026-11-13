from __future__ import annotations

import json
from pathlib import Path

import pytest

from cir_qdiag_openai_cirr.data import CIRRDataError, load_cirr_train


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "cirr"
    image_dir = root / "train"
    image_dir.mkdir(parents=True)
    for image_id in ("r0", "t0", "d0"):
        (image_dir / f"{image_id}.png").write_bytes(b"png")
    annotations = [
        {
            "pairid": 0,
            "reference": "r0",
            "target_hard": "t0",
            "caption": "  Make the red dog run  ",
            "img_set": {"id": 1, "members": ["r0", "t0", "d0"]},
        }
    ]
    image_map = {
        image_id: f"./train/{image_id}.png"
        for image_id in ("r0", "t0", "d0")
    }
    annotation_path = tmp_path / "cap.rc2.train.json"
    split_path = tmp_path / "split.rc2.train.json"
    annotation_path.write_text(json.dumps(annotations), encoding="utf-8")
    split_path.write_text(json.dumps(image_map), encoding="utf-8")
    return annotation_path, split_path, root


def test_loader_builds_official_cirr_train_sample(tmp_path: Path) -> None:
    annotation_path, split_path, root = _fixture(tmp_path)
    records = load_cirr_train(annotation_path, split_path, root)

    assert len(records) == 1
    record = records[0]
    assert record.sample_id == "cirr_train_0"
    assert record.split == "train"
    assert record.reference.image_id == "r0"
    assert record.target.image_id == "t0"
    assert record.positive_text == "Make the red dog run"
    assert record.image_set == ["r0", "t0", "d0"]
    assert Path(record.reference.image_path).is_file()


def test_loader_rejects_missing_mapped_image(tmp_path: Path) -> None:
    annotation_path, split_path, root = _fixture(tmp_path)
    (root / "train/t0.png").unlink()
    with pytest.raises(CIRRDataError, match="missing image"):
        load_cirr_train(annotation_path, split_path, root)


def test_real_cirr_train_count_and_first_row() -> None:
    root = Path("/mnt/data/datasets/search_dataset/zxl/CIR_new/datasets/CIRR")
    annotation = root / "cirr/captions/cap.rc2.train.json"
    split = root / "cirr/image_splits/split.rc2.train.json"
    if not annotation.is_file():
        return
    records = load_cirr_train(annotation, split, root)
    assert len(records) == 28225
    assert records[0].sample_id == "cirr_train_0"
    assert records[0].reference.image_id == "train-11041-2-img0"
    assert records[0].target.image_id == "train-11041-0-img1"
