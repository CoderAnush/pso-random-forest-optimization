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

Implemented and run. The full experiment (3 datasets × 5 outer folds × {default RF, random search, PSO}) is in
`results/20260925-083324_default/` (verified), with figures in `plots/20260925-083324_default/` and the write-up in
[report/REPORT.md](report/REPORT.md). Headline: PSO **maintains** the default Random Forest's test accuracy and
ties an equal-budget random search on this 3-variable space, while needing 10–24% fewer model fits.

## Installation

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

## Interactive frontend: Swarm Lab

```
.venv/Scripts/python -m pso_rf web           # opens http://localhost:8600
```

Five views:
- **Live lab:** tune N, T, w, c₁, c₂, the velocity clamp, fold and seed, then watch the real closed loop in 3-D while it races random search. The sealed test fold opens only at the end.
- **Playground:** instant PSO over *measured* Random Forest accuracy landscapes, in 2-D map and 3-D terrain views, with presets and a feedback ablation.
- **Results:** the verified experiment.
- **Replay:** saved runs in 3-D.
- **How it works:** the loop, the protocol, and live proofs.

Works offline (three.js is bundled).

Classic Streamlit demo:

```
.venv/Scripts/python -m pso_rf demo          # or: .venv/Scripts/python -m streamlit run app.py
```

Four views: a **live closed-loop lab** (watch PSO ⇄ Random Forest close the loop, race random search, open the
sealed test fold once), the **experiment results** dashboard, a **swarm replay**, and **how it works** (with a
live test-isolation proof and a feedback ablation).

## Usage

| Command | Purpose |
|---|---|
| `python -m pytest -q` | run the test suite (unit, integration, experiment level; about 3 minutes) |
| `python -m pso_rf run --config configs/default.yaml` | full experiment: 3 datasets × 5 outer folds × {baseline, random search, PSO}, plus a deployment run per dataset |
| `python -m pso_rf run --config configs/default.yaml --config configs/demo.yaml` | small-budget live demo (Iris, fold 0) with a per-iteration trace |
| `python -m pso_rf verify --results results/<exp_id>` | audit a results directory: completeness, test isolation, summary consistency |
| `python -m pso_rf compare <results_a> <results_b>` | check two runs are identical apart from timing fields |
| `python -m pso_rf plot --results results/<exp_id>` | figures F3–F11 generated from the saved result files |

Useful `run` options: `--datasets iris digits`, `--folds 0 1`, `--methods baseline pso`, `--set pso.max_iter=30`.
Use the venv interpreter (`.venv/Scripts/python`) for all commands.

Results are written to `results/<exp_id>/` and figures to `plots/<exp_id>/`, as defined in
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
