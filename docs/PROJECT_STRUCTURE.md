# PROJECT STRUCTURE

The recommended repository layout, with the responsibility of every directory. **Phase 0 creates only the Markdown
documents, `README.md` and `.gitignore`.** Everything else is created in the phase named in the last column.

`CLAUDE.md` stays at the repo root because Claude Code loads it only from there; the other 13 design documents live
in `docs/`.

```
pso-random-forest-optimization/
│
├── CLAUDE.md                     master rules for development sessions               P0
├── README.md                     install, run, results location, doc index           P0 (stub) / P15
├── .gitignore                                                                        P0
├── .gitattributes                keeps data/raw/ byte-exact (no EOL conversion)       P2
│
├── docs/
│   ├── IDEA.md                   project idea (faculty-facing)                       P0
│   ├── CONTEXT.md                technical reference: confirmed / proposed / verify  P0
│   ├── ARCHITECTURE.md           architecture, sections A–M                          P0
│   ├── REQUIREMENTS.md           ID'd requirements with traceability                 P0
│   ├── MATHEMATICAL_FORMULATION.md  problem, PSO equations, final evaluation         P0
│   ├── EXPERIMENT_PLAN.md        protocol, metrics, plots, tables                    P0
│   ├── IMPLEMENTATION_PLAN.md    phases 0–15                                         P0
│   ├── PROJECT_STRUCTURE.md      this file                                           P0
│   ├── TESTING_STRATEGY.md       unit / integration / experiment tests               P0
│   ├── RESULTS_SCHEMA.md         result file formats                                 P0
│   ├── DECISIONS.md              architecture decision records                       P0
│   ├── REVIEW_1_MAPPING.md       faculty Review 1 → project evidence                 P0
│   └── REVIEW_2_MAPPING.md       faculty Review 2 → project evidence                 P0
│
├── pyproject.toml                package metadata, pytest and ruff config            P1
├── requirements.txt              pinned dependencies                                 P1
│
├── ppt/
│   └── CB.EN.U4ELC23005_ANUSH_RAMESH_PPT.pdf   original proposal (source of truth)    existing
│
├── configs/
│   ├── default.yaml              full default experiment                             P1
│   ├── demo.yaml                 tiny budget: live demo and smoke test               P1
│   └── test.yaml                 minimal budget used by the test suite               P1
│
├── data/
│   ├── raw/
│   │   ├── processed.cleveland.data        UCI Cleveland raw file (~18 KB, CC BY 4.0) P2
│   │   └── MANIFEST.json                   URL, download date, SHA-256, citation      P2
│   └── DATASET_AUDIT.md                    generated audit: shapes, classes, missing, duplicates  P2
│
├── scripts/
│   ├── download_cleveland.py     one-time download + checksum → data/raw/            P2
│   ├── audit_datasets.py         writes data/DATASET_AUDIT.md (until the P10 CLI)     P2
│   └── benchmark_eval.py         timing gate: t_eval and projected runtime            P6
│
├── src/
│   └── pso_rf/
│       ├── __init__.py
│       ├── __main__.py           `python -m pso_rf` → experiments.cli               P10
│       ├── datasets/
│       │   ├── __init__.py       registry: name → loader
│       │   ├── base.py           DatasetBundle
│       │   ├── iris.py
│       │   ├── digits.py
│       │   ├── heart_cleveland.py
│       │   └── audit.py          class counts, duplicates, missing, balance gate      P2
│       ├── preprocessing/
│       │   ├── __init__.py
│       │   └── pipeline.py       PreprocessingSpec → sklearn steps (imputer)          P3
│       ├── models/
│       │   ├── __init__.py
│       │   ├── random_forest.py  build_model(config, seed, preprocessing) → Pipeline  P5
│       │   └── baseline.py       BASELINE_CONFIG (sklearn defaults)                   P5
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── splits.py         outer_folds, OptimizationData, HeldOutTestSet,       P4
│       │   │                     OptimizationPhase, TestSetAccessError
│       │   ├── fitness.py        FitnessEvaluator (+cache), FitnessResult             P6
│       │   ├── metrics.py        test metric computation                              P11
│       │   └── final.py          final_evaluate                                       P11
│       ├── optimization/         ← NumPy + stdlib only (UT-22)
│       │   ├── __init__.py
│       │   ├── search_space.py   IntParam, SearchSpace (decode / clip / sample)       P7
│       │   ├── events.py         Evaluation, EvaluationEvent, IterationSummary,       P7
│       │   │                     OptimizationResult, Callback protocol
│       │   ├── pso.py            PSOConfig, PSOOptimizer                              P7
│       │   └── random_search.py  RandomSearch                                         P8
│       ├── experiments/
│       │   ├── __init__.py
│       │   ├── config.py         YAML load, layering, validation, frozen dataclasses  P1/P10
│       │   ├── seeding.py        seed resolution per run                              P8
│       │   ├── recorder.py       Recorder callback → CSV/JSON                         P8/P9
│       │   ├── runner.py         run_fold, run_experiment (THE loop wiring)           P8/P10
│       │   ├── summary.py        summary_folds.csv, summary.csv                       P11
│       │   ├── verify.py         results audit (ET-02..04) and run comparison (IT-08) P10
│       │   └── cli.py            run / plot / verify / compare / audit commands       P10
│       ├── visualization/
│       │   ├── __init__.py
│       │   └── plots.py          figures F3–F11 from result files + SOURCES.json      P12
│       └── utils/
│           ├── __init__.py
│           ├── hashing.py        sha256 of files, arrays, configs                     P1
│           ├── io.py             atomic JSON/CSV writes                               P1
│           ├── manifest.py       environment + git capture                            P10
│           └── log.py            logging setup with run context                       P1
│
├── tests/
│   ├── conftest.py               shared fixtures (tiny datasets, stub objectives)     P1
│   ├── unit/                     UT-01 … UT-22                                        P2–P12
│   ├── integration/              IT-01 … IT-13                                        P8–P13
│   └── experiment/               ET-01 … ET-06                                        P10–P14
│
├── experiments/                  saved experiment configurations + short notes on    P10
│   └── README.md                 why each named experiment was run
│
├── results/                      generated: results/<exp_id>/… (RESULTS_SCHEMA.md)    P10
│
├── plots/                        generated: plots/<exp_id>/*.png + SOURCES.json       P12
│
├── notebooks/                    exploration and analysis ONLY; read results files,  optional
│                                 never produce reported numbers
│
└── report/                       final report and slide assets; figures copied from  P15
                                  plots/, numbers cited from results/
```

## Directory responsibilities

| Directory | Responsibility | Rules |
|---|---|---|
| repo root `CLAUDE.md`, `README.md` | master development rules; install, run and doc index | `CLAUDE.md` must stay at the root (Claude Code loads it only from there) |
| `docs/` | the other 13 design documents | updated whenever a decision changes (DOC-002) |
| `ppt/` | original proposal PDF | read-only; the source of the original idea |
| `configs/` | experiment configurations (YAML) | every experiment is fully described by one config plus the layers |
| `data/raw/` | the Cleveland raw file and its manifest | committed; never modified; SHA-256 verified at load |
| `data/DATASET_AUDIT.md` | generated dataset facts (resolves the TO VERIFY items) | regenerated by `python scripts/audit_datasets.py` (`python -m pso_rf audit` from Phase 10); UT-13 checks it is current |
| `scripts/` | one-time utilities (download, audit report) | not imported by the package |
| `src/pso_rf/datasets/` | dataset loading and auditing | no ML fitting |
| `src/pso_rf/preprocessing/` | preprocessing step specification | steps must be pipeline members (fitted per fit) |
| `src/pso_rf/models/` | RF and baseline construction | no data splitting; no optimization |
| `src/pso_rf/evaluation/` | splits, test guard, fitness, final metrics | **must not import `optimization`** |
| `src/pso_rf/optimization/` | search space, PSO, random search | **NumPy and stdlib only**; no data access |
| `src/pso_rf/experiments/` | config, runner, recorder, CLI | the only package that wires the closed loop |
| `src/pso_rf/visualization/` | figures from saved results | reads files only; trains nothing |
| `src/pso_rf/utils/` | shared helpers | no domain logic |
| `tests/` | the automated test suite | mirrors the IDs in TESTING_STRATEGY.md |
| `experiments/` | named experiment definitions and notes | human-readable record of what was run and why |
| `results/` | experiment outputs | generated; committed for the final experiment (evidence), ignored for scratch runs (`results/scratch_*`) |
| `plots/` | figures | generated from `results/`; each has a source record |
| `notebooks/` | optional exploration | must not contain core logic or produce reported numbers |
| `report/` | final report and slide assets | every number cites a results file (DOC-005) |

## Why `src/pso_rf/` and not top-level `src/datasets/` etc.

A top-level package named `datasets` would clash with the widely installed HuggingFace `datasets` package, and
`models` and `utils` are equally collision-prone. Namespacing under `pso_rf` avoids import ambiguity. The `src/` layout
means tests always import the installed package (`pip install -e .`), never stray files (ADR-023).

## Import layering rule (enforced by UT-22)

```
experiments ──▶ optimization        experiments ──▶ evaluation ──▶ models ──▶ preprocessing
experiments ──▶ datasets            visualization ──▶ utils        (everything) ──▶ utils
optimization ──▶ (numpy, stdlib)    ✗ optimization must not import evaluation/models/datasets/sklearn
                                    ✗ evaluation must not import optimization
```

## Git policy

- `.gitignore` excludes `__pycache__/`, `*.pyc`, `.venv/`, `venv/`, `.pytest_cache/`, `.ipynb_checkpoints/`,
  `results/scratch_*/`, `plots/scratch_*/`, `*.egg-info/`, `.ruff_cache/`, `build/` and `dist/`.
- The final experiment's `results/<exp_id>/` and `plots/<exp_id>/` **are committed**, since they are the evidence for
  Review 2.
- `data/raw/processed.cleveland.data` is committed (CC BY 4.0, attribution in `MANIFEST.json`).
