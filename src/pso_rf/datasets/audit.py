"""Dataset audit: shape, class balance, missing values, duplicates and the class-ratio gate (DR-005, DR-006).

The audit only counts; it fits nothing (DR-010).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from pso_rf.datasets.base import DatasetBundle

DUPLICATE_REMOVAL_THRESHOLD = 0.01  # exact duplicates are removed only above 1% of rows (DR-005)


def select_fitness_metric(class_ratio: float, class_ratio_gate: float | None, configured_metric: str) -> str:
    """Apply the class-ratio gate (ADR-006).

    A max/min class ratio above the gate selects ``balanced_accuracy``; otherwise, or when the gate is
    disabled (``None``), the configured metric is kept.
    """
    if class_ratio_gate is not None and class_ratio > class_ratio_gate:
        return "balanced_accuracy"
    return configured_metric


def count_duplicate_rows(X: np.ndarray, y: np.ndarray) -> int:
    """Rows whose (X, y) exactly repeats an earlier row; NaNs compare equal (pandas ``duplicated``)."""
    frame = pd.DataFrame(X)
    frame["__label__"] = y
    return int(frame.duplicated().sum())


def describe(bundle: DatasetBundle) -> dict[str, Any]:
    """Gate-independent facts: sizes, class counts and ratio, missing values per feature, duplicate rows."""
    n_samples, n_features = bundle.X.shape
    counts = np.bincount(bundle.y, minlength=len(bundle.class_names))
    if counts.min() == 0:
        raise ValueError(f"{bundle.name}: class {int(np.argmin(counts))} has no samples")
    missing = np.isnan(bundle.X).sum(axis=0)
    return {
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "n_classes": len(bundle.class_names),
        "class_counts": {str(label): int(count) for label, count in enumerate(counts)},
        "class_ratio": float(counts.max() / counts.min()),
        "missing_per_feature": {name: int(m) for name, m in zip(bundle.feature_names, missing, strict=True)},
        "n_duplicate_rows": count_duplicate_rows(bundle.X, bundle.y),
    }


def audit(
    bundle: DatasetBundle, class_ratio_gate: float | None = 1.5, configured_metric: str = "accuracy"
) -> dict[str, Any]:
    """Return the ``audit.json`` record for one dataset (RESULTS_SCHEMA §4), fields in schema order.

    Raises NotImplementedError if exact duplicates exceed 1% of rows: DR-005 would then require removing
    them before splitting, and no dataset in this project reaches that threshold.
    """
    facts = describe(bundle)
    n_duplicates, n_samples = facts["n_duplicate_rows"], facts["n_samples"]
    remove = n_duplicates > DUPLICATE_REMOVAL_THRESHOLD * n_samples
    if remove:
        raise NotImplementedError(
            f"{bundle.name}: {n_duplicates} exact duplicate rows exceed 1% of {n_samples}; "
            "DR-005 requires removing them before splitting, which is not implemented"
        )
    return {
        "dataset": bundle.name,
        "n_samples": n_samples,
        "n_features": facts["n_features"],
        "n_classes": facts["n_classes"],
        "class_counts": facts["class_counts"],
        "class_ratio": facts["class_ratio"],
        "fitness_metric": select_fitness_metric(facts["class_ratio"], class_ratio_gate, configured_metric),
        "missing_per_feature": facts["missing_per_feature"],
        "n_duplicate_rows": n_duplicates,
        "duplicates_removed": remove,
        "feature_names": list(bundle.feature_names),
        "class_names": list(bundle.class_names),
    }


def _missing_text(missing: Mapping[str, int]) -> str:
    cells = {name: count for name, count in missing.items() if count}
    if not cells:
        return "none"
    listed = ", ".join(f"`{name}`: {count}" for name, count in cells.items())
    return f"{listed} ({sum(cells.values())} cells)"


def _names_text(names: Sequence[str]) -> str:
    if len(names) <= 16:
        return ", ".join(names)
    return f"{', '.join(names[:3])}, …, {names[-1]} ({len(names)} names)"


def _duplicates_text(record: Mapping[str, Any]) -> str:
    if record["n_duplicate_rows"] == 0:
        return "0"
    return f"{record['n_duplicate_rows']} ({'removed' if record['duplicates_removed'] else 'kept'})"


def _summary_row(record: Mapping[str, Any]) -> str:
    cells = [
        f"`{record['dataset']}`",
        record["n_samples"],
        record["n_features"],
        record["n_classes"],
        " / ".join(str(count) for count in record["class_counts"].values()),
        f"{record['class_ratio']:.4f}",
        record["fitness_metric"],
        sum(record["missing_per_feature"].values()),
        _duplicates_text(record),
    ]
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def _detail_rows(record: Mapping[str, Any], meta: Mapping[str, Any], gate: str) -> list[tuple[str, str]]:
    classes = "; ".join(
        f"{label} = {name} ({record['class_counts'][str(label)]})"
        for label, name in enumerate(record["class_names"])
    )
    raw_sha = f"`{meta['sha256']}`" if meta.get("sha256") else "— (bundled with scikit-learn)"
    array_sha = f"`{meta['sha256_arrays']}`" if meta.get("sha256_arrays") else "—"
    return [
        ("Source", meta.get("source", "—")),
        ("Citation", meta.get("citation", "—")),
        ("Raw-file SHA-256", raw_sha),
        ("Array SHA-256 (`X.tobytes() + y.tobytes()`)", array_sha),
        ("Samples × features", f"{record['n_samples']} × {record['n_features']}"),
        ("Classes: label = name (count)", classes),
        ("Max/min class ratio", f"{record['class_ratio']:.4f}"),
        (f"Fitness metric (gate {gate})", record["fitness_metric"]),
        ("Missing values", _missing_text(record["missing_per_feature"])),
        ("Exact duplicate (X, y) rows", f"{_duplicates_text(record)}; removal only above 1% of rows"),
        ("Features", _names_text(record["feature_names"])),
    ]


def render_audit_markdown(
    audits: Sequence[Mapping[str, Any]],
    metas: Mapping[str, Mapping[str, Any]] | None = None,
    class_ratio_gate: float | None = 1.5,
) -> str:
    """Render the audits as the ``data/DATASET_AUDIT.md`` document (deterministic: no timestamps)."""
    metas = metas or {}
    gate = "disabled" if class_ratio_gate is None else f"{class_ratio_gate:g}"
    versions = f"numpy {np.__version__}, pandas {pd.__version__}, scikit-learn {sklearn.__version__}"
    lines = [
        "# Dataset audit",
        "",
        "Generated by `python scripts/audit_datasets.py` from the loaders in `src/pso_rf/datasets/`",
        f"({versions}). Do not edit by hand.",
        "",
        "The audit only counts; nothing is fitted (DR-010). Exact duplicate (X, y) rows are removed only",
        "above 1% of rows (DR-005). The fitness metric switches to balanced accuracy only above a max/min",
        f"class ratio of {gate} (DR-006).",
        "",
        "## Summary",
        "",
        "| Dataset | Samples | Features | Classes | Class counts | Max/min ratio | Fitness metric "
        "| Missing cells | Duplicate rows |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    lines += [_summary_row(record) for record in audits]
    for record in audits:
        rows = _detail_rows(record, metas.get(record["dataset"], {}), gate)
        lines += ["", f"## `{record['dataset']}`", "", "| Property | Value |", "|---|---|"]
        lines += [f"| {key} | {value} |" for key, value in rows]
    return "\n".join(lines) + "\n"


def write_audit_markdown(
    audits: Sequence[Mapping[str, Any]],
    path: Path | str,
    metas: Mapping[str, Mapping[str, Any]] | None = None,
    class_ratio_gate: float | None = 1.5,
) -> str:
    """Write :func:`render_audit_markdown` output to ``path`` (UTF-8, ``\\n`` endings); return the text."""
    text = render_audit_markdown(audits, metas, class_ratio_gate)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return text
