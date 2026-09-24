"""Tests for the utilities: hashing, atomic JSON, CSV appends and the run-context logging."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pytest

from pso_rf.utils.hashing import canonical_json, sha256_arrays, sha256_file, sha256_json
from pso_rf.utils.io import append_csv_rows, format_csv_value, write_json_atomic
from pso_rf.utils.log import RunLoggerAdapter, close_logging, run_prefix, setup_logging


def test_sha256_json_is_canonical() -> None:
    a = {"b": 1, "a": [1, 2.5], "c": {"y": None, "x": True}}
    b = {"c": {"x": True, "y": None}, "a": [1, 2.5], "b": 1}
    assert canonical_json(a) == '{"a":[1,2.5],"b":1,"c":{"x":true,"y":null}}'
    assert sha256_json(a) == sha256_json(b) == hashlib.sha256(canonical_json(a).encode("utf-8")).hexdigest()
    assert sha256_json({"a": 1}) != sha256_json({"a": 2})
    with pytest.raises(ValueError):
        sha256_json({"x": float("nan")})


def test_sha256_arrays_and_file(tmp_path: Path) -> None:
    X = np.arange(6, dtype=np.float64).reshape(3, 2)
    y = np.array([0, 1, 0], dtype=np.int64)
    assert sha256_arrays(X, y) == hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest()
    assert sha256_arrays(X, y) != sha256_arrays(X, np.array([1, 0, 0], dtype=np.int64))
    assert sha256_arrays(np.asfortranarray(X), y) == sha256_arrays(X, y)  # C-order bytes for any layout
    path = tmp_path / "raw.bin"
    path.write_bytes(b"63.0,1.0\r\n")
    assert sha256_file(path) == hashlib.sha256(b"63.0,1.0\r\n").hexdigest()


def test_write_json_atomic_replaces_and_leaves_no_temp_files(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "record.json"
    write_json_atomic(target, {"a": 1, "b": [1.5, None]})
    write_json_atomic(target, {"a": 2})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 2}
    assert b"\r\n" not in target.read_bytes()
    assert [p.name for p in target.parent.iterdir()] == ["record.json"]
    with pytest.raises(TypeError):
        write_json_atomic(target, {"bad": object()})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 2}
    assert [p.name for p in target.parent.iterdir()] == ["record.json"]


def test_format_csv_value() -> None:
    assert format_csv_value(None) == ""
    assert format_csv_value(True) == "true"
    assert format_csv_value(np.bool_(False)) == "false"
    assert format_csv_value(0.1 + 0.2) == "0.30000000000000004"  # full precision
    assert format_csv_value(float("-inf")) == "-inf"
    assert format_csv_value(np.float64(0.25)) == "0.25"
    assert format_csv_value(np.int64(7)) == "7"
    assert format_csv_value([0.5, 1.0]) == "[0.5,1.0]"
    assert format_csv_value("fold_0") == "fold_0"


def test_append_csv_rows_writes_the_header_once_in_fixed_order(tmp_path: Path) -> None:
    path = tmp_path / "table.csv"
    columns = ["a", "b", "c"]
    assert append_csv_rows(path, [{"c": None, "b": True, "a": 1}], columns) == 1
    assert (
        append_csv_rows(path, [{"a": 2, "b": False, "c": [0.5]}, {"b": True, "c": 1.5, "a": 3}], columns) == 2
    )
    assert path.read_bytes().decode("utf-8") == "a,b,c\n1,true,\n2,false,[0.5]\n3,true,1.5\n"


def test_append_csv_rows_rejects_schema_drift(tmp_path: Path) -> None:
    path = tmp_path / "table.csv"
    append_csv_rows(path, [{"a": 1, "b": 2}], ["a", "b"])
    with pytest.raises(ValueError, match="missing"):
        append_csv_rows(path, [{"a": 1}], ["a", "b"])
    with pytest.raises(ValueError, match="header"):
        append_csv_rows(path, [{"a": 1, "c": 2}], ["a", "c"])
    with pytest.raises(ValueError, match="duplicate"):
        append_csv_rows(tmp_path / "other.csv", [], ["a", "a"])


def test_run_logger_adapter_writes_the_context_prefix_to_run_log(tmp_path: Path) -> None:
    log_path = tmp_path / "run.log"
    logger = setup_logging(log_path)
    try:
        run = RunLoggerAdapter(logging.getLogger("pso_rf.test"), "iris", 0, "pso", 0)
        run.info("iter 1/20")
        RunLoggerAdapter(logging.getLogger("pso_rf.test"), "iris", None, "pso", 5).debug("evaluation 3")
        setup_logging(log_path)  # a second setup replaces the handlers instead of duplicating them
        run.info("after re-setup")
        assert sum(isinstance(h, logging.FileHandler) for h in logger.handlers) == 1
    finally:
        close_logging()
    text = log_path.read_text(encoding="utf-8")
    assert "[iris|fold 0|pso|seed 0] iter 1/20" in text
    assert "[iris|deployment|pso|seed 5] evaluation 3" in text  # DEBUG reaches run.log
    assert text.count("after re-setup") == 1
    assert (
        run_prefix("heart_cleveland", 4, "random_search", 4)
        == "[heart_cleveland|fold 4|random_search|seed 4]"
    )


def test_console_shows_info_and_warnings_but_not_debug(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging(None)
    try:
        log = logging.getLogger("pso_rf.test")
        log.debug("hidden detail")
        log.info("visible trace")
        log.warning("careful")
    finally:
        close_logging()
    err = capsys.readouterr().err
    assert "visible trace" in err
    assert "hidden detail" not in err
    assert "WARNING: careful" in err
