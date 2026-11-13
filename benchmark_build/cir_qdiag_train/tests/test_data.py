from pathlib import Path

from PIL import Image

from cir_qdiag_train.data import CIRRHardNegativeDataset
from cir_qdiag_train.manifest import (
    HardNegativeText,
    TrainManifestRecord,
)


def _negative(identifier: str, operation: str) -> HardNegativeText:
    return HardNegativeText(
        negative_id=identifier,
        text=f"negative {identifier}",
        semantic_target="object",
        edit_operation=operation,
        construction_mode="text_only_editable",
    )


def test_delete_ablation_preserves_same_non_delete_set(tmp_path: Path) -> None:
    reference = tmp_path / "reference.png"
    target = tmp_path / "target.png"
    Image.new("RGB", (8, 8)).save(reference)
    Image.new("RGB", (8, 8)).save(target)
    record = TrainManifestRecord(
        sample_id="cirr_train_0",
        pair_id=0,
        reference_id="r",
        reference_path=str(reference),
        target_id="t",
        target_path=str(target),
        positive_text="Show a cat",
        negatives=[
            _negative("add-1", "add"),
            _negative("add-2", "add"),
            _negative("sub-1", "substitute"),
            _negative("delete-1", "delete"),
        ],
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(record.model_dump_json() + "\n", encoding="utf-8")
    kwargs = {
        "manifest_path": manifest,
        "max_non_delete_negatives_per_sample": 2,
        "max_delete_negatives_per_sample": 1,
        "seed": 42,
    }
    with_delete = CIRRHardNegativeDataset(
        negative_mode="add_substitute_delete", **kwargs
    )
    without_delete = CIRRHardNegativeDataset(
        negative_mode="add_substitute", **kwargs
    )
    baseline = CIRRHardNegativeDataset(negative_mode="none", **kwargs)
    with_ids = [item.negative_id for item in with_delete[0]["negatives"]]
    without_ids = [item.negative_id for item in without_delete[0]["negatives"]]
    assert without_ids == [item for item in with_ids if not item.startswith("delete")]
    assert len(with_ids) == len(without_ids) + 1
    assert baseline[0]["negatives"] == []


def test_image_set_candidates_exclude_reference_and_target(tmp_path: Path) -> None:
    paths = {}
    for image_id in ("r", "t", "c1", "c2"):
        path = tmp_path / f"{image_id}.png"
        Image.new("RGB", (8, 8)).save(path)
        paths[image_id] = path
    record = TrainManifestRecord(
        sample_id="cirr_train_1",
        pair_id=1,
        reference_id="r",
        reference_path=str(paths["r"]),
        target_id="t",
        target_path=str(paths["t"]),
        positive_text="Show a dog",
        image_set=["r", "t", "c1", "c2"],
        negatives=[_negative("object-add", "add")],
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(record.model_dump_json() + "\n", encoding="utf-8")
    split = tmp_path / "split.json"
    split.write_text(
        "{" + ",".join(
            f'"{image_id}":"{path.name}"' for image_id, path in paths.items()
        ) + "}",
        encoding="utf-8",
    )
    dataset = CIRRHardNegativeDataset(
        manifest,
        negative_mode="add_substitute",
        max_non_delete_negatives_per_sample=1,
        max_delete_negatives_per_sample=1,
        seed=42,
        candidate_image_mode="image_set",
        train_image_split_path=split,
        image_root=tmp_path,
        max_candidate_images_per_sample=2,
    )
    example = dataset[0]
    assert sorted(example["candidate_ids"]) == ["c1", "c2"]
    assert len(example["candidate_images"]) == 2
