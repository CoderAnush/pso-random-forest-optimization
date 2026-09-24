"""Audit the three datasets and write ``data/DATASET_AUDIT.md`` (Phase 2).

Usage:
    python scripts/audit_datasets.py [--config CONFIG] [--data-dir DATA_DIR] [--out OUT]

The class-ratio gate and the configured fitness metric come from the configuration (per dataset;
default ``configs/default.yaml``). The ``audit`` command of the package CLI (Phase 10) supersedes
this script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pso_rf.datasets import LOADERS, load_dataset
from pso_rf.datasets.audit import audit, write_audit_markdown
from pso_rf.experiments.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    """Load and audit every registered dataset, write the Markdown report and print a summary."""
    parser = argparse.ArgumentParser(description="Audit the datasets and write data/DATASET_AUDIT.md.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=None, help="default: <data-dir>/DATASET_AUDIT.md")
    args = parser.parse_args(argv)

    config = load_config([args.config])
    audits, metas = [], {}
    for name in LOADERS:
        fitness = config.for_dataset(name).fitness
        bundle = load_dataset(name, args.data_dir)
        audits.append(audit(bundle, fitness.class_ratio_gate, fitness.metric))
        metas[name] = bundle.meta
    out = args.out or args.data_dir / "DATASET_AUDIT.md"
    write_audit_markdown(audits, out, metas, config.fitness.class_ratio_gate)
    for record in audits:
        missing = {key: value for key, value in record["missing_per_feature"].items() if value}
        print(
            f"{record['dataset']}: {record['n_samples']} x {record['n_features']}, "
            f"classes {record['class_counts']}, ratio {record['class_ratio']:.4f}, "
            f"metric {record['fitness_metric']}, missing {missing or 'none'}, "
            f"duplicates {record['n_duplicate_rows']}"
        )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
