"""File writing helpers: atomic JSON records and append-only CSV tables (RESULTS_SCHEMA conventions)."""

from __future__ import annotations

import contextlib
import csv
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def write_json_atomic(path: Path | str, obj: Any, *, indent: int = 2) -> None:
    """Write ``obj`` as UTF-8 JSON via a temp file and ``os.replace``, so readers never see a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=indent, ensure_ascii=False) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise


def format_csv_value(value: Any) -> str:
    """Format one CSV cell: empty for None, ``true``/``false`` for bools, ``repr`` for floats, JSON for lists.

    ``repr`` keeps full float precision and writes ``-inf`` as ``-inf`` (RESULTS_SCHEMA conventions).
    """
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, (float, np.floating)):
        return repr(float(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def append_csv_rows(path: Path | str, rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> int:
    """Append ``rows`` to a CSV file in the fixed ``columns`` order; the header is written once.

    Every row must have exactly the keys in ``columns``, and an existing file must already carry that header.
    Files are UTF-8 with ``\\n`` line endings. Returns the number of rows written.
    """
    path = Path(path)
    columns = list(columns)
    if len(set(columns)) != len(columns):
        raise ValueError(f"duplicate column names in {columns}")
    rows = list(rows)
    expected = set(columns)
    for index, row in enumerate(rows):
        if set(row) != expected:
            missing = sorted(expected - set(row))
            extra = sorted(set(row) - expected)
            raise ValueError(
                f"row {index} does not match the columns of {path.name}: missing {missing}, extra {extra}"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    if not new_file:
        with path.open("r", encoding="utf-8", newline="") as fh:
            header = next(csv.reader(fh), [])
        if header != columns:
            raise ValueError(f"{path} has header {header}, expected {columns}")

    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        if new_file:
            writer.writerow(columns)
        for row in rows:
            writer.writerow([format_csv_value(row[column]) for column in columns])
    return len(rows)


def utc_timestamp() -> str:
    """The current time as ISO-8601 UTC with milliseconds, e.g. ``2026-10-01T14:30:00.123Z``."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
