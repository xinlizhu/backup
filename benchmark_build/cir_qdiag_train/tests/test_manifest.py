import json
from pathlib import Path

import pytest
from PIL import Image

from cir_qdiag_train.manifest import build_train_manifest, read_manifest


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(100, 120, 140)).save(path)


def _dataset_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "images"
    for image_id in ("train-r0", "train-t0", "train-r1", "train-t1"):
        _write_image(root / "train" / f"{image_id}.png")
    annotations = [
        {
            "pairid": 0,
            "reference": "train-r0",
            "target_hard": "train-t0",
            "caption": "Make the cat red",
            "img_set": {"members": ["train-r0", "train-t0"]},
        },
        {
            "pairid": 1,
            "reference": "train-r1",
            "target_hard": "train-t1",
            "caption": "Put the dog on grass",
            "img_set": {"members": ["train-r1", "train-t1"]},
        },
    ]
    image_map = {
        image_id: f"./train/{image_id}.png"
        for image_id in ("train-r0", "train-t0", "train-r1", "train-t1")
    }
    annotation_path = tmp_path / "cap.json"
    split_path = tmp_path / "split.json"
    annotation_path.write_text(json.dumps(annotations), encoding="utf-8")
    split_path.write_text(json.dumps(image_map), encoding="utf-8")
    return annotation_path, split_path, root


def test_build_manifest_joins_train_negatives_and_allows_partial_coverage(
    tmp_path: Path,
) -> None:
    annotation_path, split_path, image_root = _dataset_files(tmp_path)
    negative_path = tmp_path / "negative.jsonl"
    negative_path.write_text(
        json.dumps(
            {
                "sample_id": "cirr_train_0",
                "split": "train",
                "pair_id": 0,
                "reference": {"image_id": "train-r0"},
                "target": {"image_id": "train-t0"},
                "positive_text": "Make the cat red",
                "negatives": [
                    {
                        "negative_id": "n-add",
                        "text": "Make the cat red with a dog",
                        "semantic_target": "object",
                        "edit_operation": "add",
                        "construction_mode": "text_only_editable",
                        "status": "accepted",
                    },
                    {
                        "negative_id": "n-delete",
                        "text": "Make the cat",
                        "semantic_target": "attribute",
                        "edit_operation": "delete",
                        "construction_mode": "text_only_editable",
                        "status": "accepted",
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "manifest.jsonl"
    summary = build_train_manifest(
        annotation_path=annotation_path,
        image_split_path=split_path,
        image_root=image_root,
        negative_jsonl_paths=[negative_path],
        output_path=output,
    )
    records = read_manifest(output)
    assert len(records) == 2
    assert len(records[0].negatives) == 2
    assert records[1].negatives == []
    assert summary["samples_with_negatives"] == 1
    assert summary["negatives_by_operation"] == {"add": 1, "delete": 1}


def test_manifest_rejects_validation_negatives(tmp_path: Path) -> None:
    annotation_path, split_path, image_root = _dataset_files(tmp_path)
    negative_path = tmp_path / "negative.jsonl"
    negative_path.write_text(
        json.dumps(
            {
                "sample_id": "cirr_val_0",
                "split": "val",
                "pair_id": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="never train with val output"):
        build_train_manifest(
            annotation_path=annotation_path,
            image_split_path=split_path,
            image_root=image_root,
            negative_jsonl_paths=[negative_path],
            output_path=tmp_path / "manifest.jsonl",
        )


def test_build_manifest_can_keep_only_generated_samples(tmp_path: Path) -> None:
    annotation_path, split_path, image_root = _dataset_files(tmp_path)
    negative_path = tmp_path / "negative.jsonl"
    negative_path.write_text(
        json.dumps(
            {
                "sample_id": "cirr_train_0",
                "split": "train",
                "pair_id": 0,
                "reference": {"image_id": "train-r0"},
                "target": {"image_id": "train-t0"},
                "positive_text": "Make the cat red",
                "negatives": [
                    {
                        "negative_id": "n-add",
                        "text": "Make the cat red with a dog",
                        "semantic_target": "object",
                        "edit_operation": "add",
                        "construction_mode": "text_only_editable",
                        "status": "accepted",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "manifest.jsonl"
    summary = build_train_manifest(
        annotation_path=annotation_path,
        image_split_path=split_path,
        image_root=image_root,
        negative_jsonl_paths=[negative_path],
        output_path=output,
        only_samples_with_negatives=True,
    )
    records = read_manifest(output)
    assert [record.pair_id for record in records] == [0]
    assert summary["num_train_samples"] == 1
    assert summary["only_samples_with_negatives"] is True
