from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def read_jsonl(
    path: str | Path,
    *,
    tolerate_truncated_last_line: bool = False,
) -> Iterator[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                is_truncated_tail = (
                    tolerate_truncated_last_line
                    and line_number == len(lines)
                    and not line.endswith("\n")
                )
                if is_truncated_tail:
                    return
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc


def read_models(path: str | Path, model_type: type[T]) -> Iterator[T]:
    for item in read_jsonl(path):
        yield model_type.model_validate(item)


def write_jsonl_atomic(path: str | Path, rows: Iterable[dict[str, Any] | BaseModel]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            value = row.model_dump(mode="json") if isinstance(row, BaseModel) else row
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


class ResumableJsonlWriter:
    def __init__(
        self,
        path: str | Path,
        id_field: str = "sample_id",
        force: bool = False,
        *,
        order_ids: Iterable[str | int] | None = None,
        retry_predicate: Callable[[dict[str, Any]], bool] | None = None,
    ):
        self.path = Path(path)
        self.id_field = id_field
        self.retry_predicate = retry_predicate or (lambda _: False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if force and self.path.exists():
            self.path.unlink()
        self.order_ids = [str(item_id) for item_id in order_ids] if order_ids is not None else None
        if self.order_ids is not None and len(self.order_ids) != len(set(self.order_ids)):
            raise ValueError("order_ids must be unique")
        self._order_set = set(self.order_ids) if self.order_ids is not None else None
        self.rows: dict[str, dict[str, Any]] = {}
        repair_tail = bool(
            self.path.exists() and self.path.stat().st_size and not _ends_with_newline(self.path)
        )
        if self.path.exists():
            for item in read_jsonl(self.path, tolerate_truncated_last_line=True):
                if id_field not in item:
                    raise ValueError(f"JSONL row is missing id field {id_field!r}")
                self.rows[str(item[id_field])] = item
        if self.order_ids is not None:
            extra = set(self.rows) - self._order_set
            if extra:
                raise ValueError(
                    "existing output contains ids outside the current input; "
                    f"use --force to rebuild: {sorted(extra)}"
                )
        if repair_tail:
            # A crash can leave a partial final journal record. Compact the valid prefix before
            # appending so another crash cannot turn that fragment into an invalid middle line.
            write_jsonl_atomic(self.path, self.rows.values())
        self.completed = {
            item_id for item_id, item in self.rows.items() if not self.retry_predicate(item)
        }
        self._handle = self.path.open("a", encoding="utf-8")

    def has(self, item_id: str | int) -> bool:
        return str(item_id) in self.completed

    def get(self, item_id: str | int) -> dict[str, Any] | None:
        return self.rows.get(str(item_id))

    def get_model(self, item_id: str | int, model_type: type[T]) -> T | None:
        value = self.get(item_id)
        return None if value is None else model_type.model_validate(value)

    def append(self, row: dict[str, Any] | BaseModel) -> None:
        value = row.model_dump(mode="json") if isinstance(row, BaseModel) else row
        item_id = str(value[self.id_field])
        if self._order_set is not None and item_id not in self._order_set:
            raise ValueError(f"cannot append id outside order_ids: {item_id}")
        self._handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self.rows[item_id] = value
        if self.retry_predicate(value):
            self.completed.discard(item_id)
        else:
            self.completed.add(item_id)

    def close(self) -> None:
        self._handle.close()
        order = self.order_ids or list(self.rows)
        write_jsonl_atomic(
            self.path, (self.rows[item_id] for item_id in order if item_id in self.rows)
        )

    def __enter__(self) -> "ResumableJsonlWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _ends_with_newline(path: Path) -> bool:
    with path.open("rb") as handle:
        handle.seek(-1, os.SEEK_END)
        return handle.read(1) == b"\n"
