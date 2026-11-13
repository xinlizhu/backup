from __future__ import annotations

from pathlib import Path

import pandas as pd

from cir_qdiag.data import load_exact_pinpoint_groups


def _row(query_id: str, instruction: str, *, positive=("p1", "p2"), negative=("n1",)):
    return {
        "query_id": query_id,
        "query_image_signature": "r1",
        "query_image_signature2": "None",
        "instruction": instruction,
        "positive_candidates": list(positive),
        "negative_candidates": [*negative, None, *negative],
        "token_count": len(instruction.split()),
        "length_category": "short",
        "is_question": False,
        "is_imperative": True,
        "query_category": "Swap",
        "l1_interest": "Art",
    }


def test_loader_keeps_only_exact_groups_of_two_to_six(tmp_path: Path) -> None:
    rows = [
        _row("query_00001", "Show a red dog"),
        _row("query_00002", "A red dog"),
        _row("query_00003", "This has different positives", positive=("p3",)),
    ]
    rows.extend(_row(f"query_0010{i}", f"oversized {i}", positive=("p4",)) for i in range(7))
    parquet = tmp_path / "pinpoint.parquet"
    pd.DataFrame(rows).to_parquet(parquet)

    groups = load_exact_pinpoint_groups(parquet, tmp_path / "images")

    assert len(groups) == 1
    group = groups[0]
    assert [item.instruction_id for item in group.instructions] == [
        "query_00001",
        "query_00002",
    ]
    assert [item.image_id for item in group.positive_images] == ["p1", "p2"]
    assert group.official_negative_image_ids == ["n1"]
    assert len(group.references) == 1


def test_real_pinpoint_exact_group_count() -> None:
    root = Path("/mnt/data/datasets/search_dataset/zxl/pinpoint-dataset")
    if not (root / "pinpoint_licensed.parquet").is_file():
        return
    groups = load_exact_pinpoint_groups(
        root / "pinpoint_licensed.parquet",
        root / "image_cache",
    )
    assert len(groups) == 1683
    assert sum(len(group.instructions) for group in groups) == 6553
    assert all(2 <= len(group.instructions) <= 6 for group in groups)
