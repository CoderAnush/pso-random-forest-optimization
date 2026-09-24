"""UT-13: dataset loaders (shapes, labels, Heart binarisation, '?' -> NaN, SHA-256 verification)."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import numpy as np
import pytest

from pso_rf.datasets import LOADERS, DatasetBundle, DatasetError, load_dataset
from pso_rf.datasets.audit import audit, render_audit_markdown
from pso_rf.datasets.heart_cleveland import COLUMNS, MANIFEST_FILE, RAW_FILE
from pso_rf.experiments.config import load_config
from pso_rf.utils.hashing import sha256_arrays


@pytest.mark.parametrize(
    ("name", "shape", "n_classes"),
    [("iris", (150, 4), 3), ("digits", (1797, 64), 10), ("heart_cleveland", (303, 13), 2)],
)
def test_loader_shapes_types_and_labels(
    data_dir: Path, name: str, shape: tuple[int, int], n_classes: int
) -> None:
    bundle = load_dataset(name, data_dir)
    assert bundle.name == name
    assert bundle.X.shape == shape and bundle.X.dtype == np.float64
    assert bundle.y.shape == (shape[0],) and bundle.y.dtype == np.int64
    assert len(bundle.class_names) == n_classes
    assert set(np.unique(bundle.y).tolist()) == set(range(n_classes))
    assert len(bundle.feature_names) == shape[1]
    assert bundle.meta["sha256_arrays"] == sha256_arrays(bundle.X, bundle.y)
    facts = bundle.meta["audit"]
    assert (facts["n_samples"], facts["n_features"], facts["n_classes"]) == (*shape, n_classes)
    assert list(facts["class_counts"].values()) == np.bincount(bundle.y).tolist()


def test_committed_audit_report_matches_the_loaders(data_dir: Path, configs_dir: Path) -> None:
    """``data/DATASET_AUDIT.md`` must equal a fresh audit, so every count in it matches the loaded data."""
    config = load_config([configs_dir / "default.yaml"])
    audits, metas = [], {}
    for name in LOADERS:
        bundle = load_dataset(name, data_dir)
        fitness = config.for_dataset(name).fitness
        audits.append(audit(bundle, fitness.class_ratio_gate, fitness.metric))
        metas[name] = bundle.meta
    expected = render_audit_markdown(audits, metas, config.fitness.class_ratio_gate)
    assert (data_dir / "DATASET_AUDIT.md").read_text(encoding="utf-8") == expected


def _raw_rows(data_dir: Path) -> list[list[str]]:
    with (data_dir / "raw" / RAW_FILE).open(newline="", encoding="ascii") as fh:
        return [row for row in csv.reader(fh) if row]


def test_heart_target_is_num_greater_than_zero(heart_bundle: DatasetBundle, data_dir: Path) -> None:
    num = np.array([int(float(row[-1])) for row in _raw_rows(data_dir)])
    assert set(num.tolist()) <= {0, 1, 2, 3, 4}
    assert np.array_equal(heart_bundle.y, (num > 0).astype(np.int64))
    assert heart_bundle.class_names == ["no_disease", "disease"]
    assert heart_bundle.feature_names == list(COLUMNS[:-1])


def test_heart_question_marks_become_nan_and_other_cells_are_kept(
    heart_bundle: DatasetBundle, data_dir: Path
) -> None:
    rows = _raw_rows(data_dir)
    marks = {(i, j) for i, row in enumerate(rows) for j, value in enumerate(row[:-1]) if value.strip() == "?"}
    nans = {(int(i), int(j)) for i, j in zip(*np.nonzero(np.isnan(heart_bundle.X)), strict=True)}
    assert marks, "the raw file is expected to contain '?' cells"
    assert nans == marks
    for i, row in enumerate(rows):
        for j, value in enumerate(row[:-1]):
            if (i, j) not in marks:
                assert heart_bundle.X[i, j] == float(value)


@pytest.fixture
def heart_copy(data_dir: Path, tmp_path: Path) -> Path:
    """A data directory in tmp_path holding copies of the raw file and its manifest."""
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    for name in (RAW_FILE, MANIFEST_FILE):
        shutil.copyfile(data_dir / "raw" / name, raw_dir / name)
    return tmp_path / "data"


def test_heart_loads_from_a_verified_copy(heart_copy: Path, heart_bundle: DatasetBundle) -> None:
    assert (
        load_dataset("heart_cleveland", heart_copy).meta["sha256_arrays"]
        == heart_bundle.meta["sha256_arrays"]
    )


def test_tampered_heart_file_fails_the_checksum(heart_copy: Path) -> None:
    raw = heart_copy / "raw" / RAW_FILE
    data = raw.read_bytes()
    tampered = data.replace(b"63.0,1.0,1.0", b"64.0,1.0,1.0", 1)
    assert tampered != data
    raw.write_bytes(tampered)
    with pytest.raises(DatasetError, match="SHA-256 mismatch") as info:
        load_dataset("heart_cleveland", heart_copy)
    assert "download_cleveland.py" in str(info.value)


def test_line_ending_conversion_fails_the_checksum(heart_copy: Path) -> None:
    """Why .gitattributes marks data/raw as -text: a CRLF checkout would change the bytes."""
    raw = heart_copy / "raw" / RAW_FILE
    raw.write_bytes(raw.read_bytes().replace(b"\n", b"\r\n"))
    with pytest.raises(DatasetError, match="SHA-256 mismatch"):
        load_dataset("heart_cleveland", heart_copy)


def test_missing_file_or_manifest_points_to_the_download_script(heart_copy: Path, tmp_path: Path) -> None:
    (heart_copy / "raw" / MANIFEST_FILE).unlink()
    with pytest.raises(DatasetError, match="download_cleveland.py"):
        load_dataset("heart_cleveland", heart_copy)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(DatasetError, match="download_cleveland.py"):
        load_dataset("heart_cleveland", empty)


def test_registry_and_unknown_names(data_dir: Path) -> None:
    assert list(LOADERS) == ["iris", "digits", "heart_cleveland"]
    with pytest.raises(ValueError, match="unknown dataset"):
        load_dataset("mnist", data_dir)


def test_bundle_rejects_inconsistent_arrays() -> None:
    X = np.zeros((3, 2))
    y = np.array([0, 1, 0], dtype=np.int64)
    DatasetBundle("ok", X, y, ["a", "b"], ["neg", "pos"], {})
    bad_arguments = [
        (X, y[:2], ["a", "b"], ["neg", "pos"]),  # row count mismatch
        (X.astype(np.float32), y, ["a", "b"], ["neg", "pos"]),  # X dtype
        (X, y.astype(np.int32), ["a", "b"], ["neg", "pos"]),  # y dtype
        (X, np.array([0, 1, 2], dtype=np.int64), ["a", "b"], ["neg", "pos"]),  # label outside 0..C-1
        (X, y, ["a"], ["neg", "pos"]),  # feature-name count
    ]
    for X_bad, y_bad, features, classes in bad_arguments:
        with pytest.raises(ValueError):
            DatasetBundle("bad", X_bad, y_bad, features, classes, {})
