from __future__ import annotations

import json
from pathlib import Path

import pytest

from cir_qdiag.utils.jsonl import ResumableJsonlWriter, read_jsonl


def _retry_failed(row: dict) -> bool:
    return row.get("metadata", {}).get("operational_status") == "failed"


def _row(sample_id: str, status: str, value: int) -> dict:
    return {
        "sample_id": sample_id,
        "value": value,
        "metadata": {"operational_status": status},
    }


def test_failed_rows_are_upserted_and_compacted_in_input_order(tmp_path: Path) -> None:
    path = tmp_path / "stage.jsonl"
    order = ["a", "b", "c"]
    with ResumableJsonlWriter(path, order_ids=order, retry_predicate=_retry_failed) as writer:
        writer.append(_row("a", "ok", 1))
        writer.append(_row("b", "failed", 1))
        writer.append(_row("c", "ok", 1))

    with ResumableJsonlWriter(path, order_ids=order, retry_predicate=_retry_failed) as writer:
        assert writer.has("a")
        assert not writer.has("b")
        assert writer.has("c")
        assert writer.get("b")["value"] == 1
        writer.append(_row("b", "ok", 2))

    rows = list(read_jsonl(path))
    assert [row["sample_id"] for row in rows] == order
    assert len({row["sample_id"] for row in rows}) == 3
    assert rows[1]["value"] == 2


def test_truncated_tail_is_removed_before_new_journal_writes(tmp_path: Path) -> None:
    path = tmp_path / "stage.jsonl"
    path.write_text(
        json.dumps(_row("a", "ok", 1)) + "\n" + '{"sample_id":"b"',
        encoding="utf-8",
    )
    with ResumableJsonlWriter(path, order_ids=["a", "b"]) as writer:
        assert writer.has("a")
        assert not writer.has("b")
        writer.append(_row("b", "ok", 2))

    assert [row["sample_id"] for row in read_jsonl(path)] == ["a", "b"]


def test_invalid_middle_line_is_never_silently_ignored(tmp_path: Path) -> None:
    path = tmp_path / "stage.jsonl"
    path.write_text(
        json.dumps(_row("a", "ok", 1)) + "\n{" + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid JSONL"):
        ResumableJsonlWriter(path, order_ids=["a"])


def test_writer_rejects_ids_outside_current_input(tmp_path: Path) -> None:
    path = tmp_path / "stage.jsonl"
    with ResumableJsonlWriter(path, order_ids=["a"]) as writer:
        with pytest.raises(ValueError, match="outside order_ids"):
            writer.append(_row("b", "ok", 1))
