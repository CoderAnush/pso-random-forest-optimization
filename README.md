# PSO-Based Random Forest Hyperparameter Optimization

A from-scratch Particle Swarm Optimization (PSO) tunes three Random Forest hyperparameters (`n_estimators`,
`max_depth`, `min_samples_split`) on Iris, Digits and UCI Cleveland Heart Disease as a **closed feedback loop**:
PSO proposes candidate hyperparameters, a Random Forest is trained with them, its inner stratified 5-fold
cross-validation accuracy is fed back to PSO as fitness, and PSO updates its personal and global bests, velocities
and positions to propose the next candidates (10 particles, 20 iterations). The data is split by an outer stratified
5-fold cross-validation: each outer test fold stays sealed during the loop and is evaluated **once**, after
optimization ends, and its result is never fed back. PSO is compared with the default Random Forest and with an
equal-budget random search (the open-loop control).

Evolutionary Optimization mini-project: Anush Ramesh (CB.EN.U4ELC23005), EEE, Amrita Vishwa Vidyapeetham, Coimbatore.

## Status

Implementation in progress; no results yet. The design is fixed in the documents listed below.

## Installation (available from Phase 1)

Python 3.10. The pinned dependency versions are in `requirements.txt`.

If those packages are already installed system-wide (as on the development machine), reuse them without
downloading anything:

```
python -m venv --system-site-packages .venv
.venv/Scripts/python -m pip install -e . --no-deps --no-build-isolation
```

On a clean machine, install the pinned versions first:

```
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pip install -e . --no-deps
```

(On Linux or macOS, use `.venv/bin/python`.)

## Usage (planned)

| Command | Purpose | Available from |
|---|---|---|
| `python -m pytest -q` | run the test suite | Phase 1 |
| `python -m pso_rf run --config configs/default.yaml` | full experiment: 3 datasets × 5 outer folds × {baseline, random search, PSO}, plus a deployment run per dataset | Phase 10 |
| `python -m pso_rf run --config configs/demo.yaml` | small-budget live demo (Iris, fold 0) with a per-iteration trace | Phase 10 |
| `python -m pso_rf plot --results results/<exp_id>` | figures generated from saved result files | Phase 12 |

Results will be written to `results/<exp_id>/` and figures to `plots/<exp_id>/`, as defined in
[docs/RESULTS_SCHEMA.md](docs/RESULTS_SCHEMA.md).

## Documentation

| Document | Contents |
|---|---|
| [CLAUDE.md](CLAUDE.md) | Master development rules: closed-loop and test-isolation rules, conventions, forbidden shortcuts |
| [docs/IDEA.md](docs/IDEA.md) | Project idea for the faculty: problem, why Random Forest and PSO, the feedback loop |
| [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) | Numbered requirements (FR, NFR, AR, MLR, OR, DR, ER, RR, DOC) with traceability |
| [docs/CONTEXT.md](docs/CONTEXT.md) | Technical reference: confirmed, proposed and to-verify facts; module interfaces; configuration; seeds |
| [docs/MATHEMATICAL_FORMULATION.md](docs/MATHEMATICAL_FORMULATION.md) | Search space, objective, PSO equations, integer decoding, worked example, final evaluation |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Architecture diagrams, closed-loop control mapping, test isolation, configuration template |
| [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) | Repository layout and the import-layering rule |
| [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md) | Research questions, datasets, protocol, metrics, planned plots and tables |
| [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) | Phases 0–15 with files, tests and completion criteria |
| [docs/TESTING_STRATEGY.md](docs/TESTING_STRATEGY.md) | Unit, integration and experiment tests, including the isolation and closed-loop proofs |
| [docs/RESULTS_SCHEMA.md](docs/RESULTS_SCHEMA.md) | Result files, columns and types |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Architecture decision records (ADR-001 to ADR-025) |
| [docs/REVIEW_1_MAPPING.md](docs/REVIEW_1_MAPPING.md) | Faculty Review 1 requirements mapped to the project (Review 1 completed) |
| [docs/REVIEW_2_MAPPING.md](docs/REVIEW_2_MAPPING.md) | Faculty Review 2 requirements mapped to planned evidence |

The original proposal (the source of truth for the idea) is `ppt/CB.EN.U4ELC23005_ANUSH_RAMESH_PPT.pdf`.
