"""HTTP + server-sent-events backend for the web frontend (ADR-027).

* ``/api/live``: starts a live closed-loop job. PSO (and optionally random search, concurrently) run
  through ``run_fold`` with an observer callback. Every evaluation and iteration is streamed to the
  browser; the default RF and the manual pick are scored; the test metrics of every method are released
  together at the end.
* ``/api/results``, ``/api/replay``, ``/api/landscape``: read saved result files only.
* ``/api/proof/isolation``: tries to read a held-out test set inside an optimization phase (refused).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
import time
import uuid
from datetime import datetime, timezone
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tornado.web
from tornado.iostream import StreamClosedError

from pso_rf.datasets import DatasetBundle, load_dataset
from pso_rf.datasets.audit import audit
from pso_rf.evaluation import (
    FitnessEvaluator,
    HeldOutTestSet,
    OptimizationPhase,
    TestSetAccessError,
    outer_folds,
)
from pso_rf.evaluation.final import final_evaluate
from pso_rf.experiments.config import ExperimentConfig, load_config
from pso_rf.experiments.runner import positive_label_for, run_fold
from pso_rf.experiments.verify import verify_results
from pso_rf.optimization import Callback, EvaluationEvent, IterationSummary

REPO = Path(__file__).resolve().parents[3]
STATIC = Path(__file__).resolve().parent / "static"
RESULTS = REPO / "results"
HP = ("n_estimators", "max_depth", "min_samples_split")
LABELS = {"iris": "Iris", "digits": "Digits", "heart_cleveland": "Heart Disease"}
TASKS = {
    "iris": "3-class flower species",
    "digits": "10-class handwritten digits (8×8 pixels)",
    "heart_cleveland": "binary heart-disease diagnosis (UCI Cleveland)",
}
LIMITS = {  # clamp ranges for user-supplied live parameters
    "n_particles": (2, 30, int),
    "max_iter": (1, 40, int),
    "w": (0.05, 1.2, float),
    "c1": (0.0, 3.0, float),
    "c2": (0.0, 3.0, float),
    "v_max_frac": (0.02, 1.0, float),
    "fold": (0, 4, int),
    "seed": (0, 1_000_000, int),
}
MAX_ACTIVE_JOBS = 2
log = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------------------ helpers


def clean(value: Any) -> Any:
    """JSON-safe copy: non-finite floats → None, numpy scalars/arrays → Python types."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@cache
def bundle(name: str) -> DatasetBundle:
    return load_dataset(name, REPO / "data")


def experiments() -> list[dict[str, Any]]:
    """Saved experiments (newest first) with their completion status."""
    found = []
    for root in sorted(RESULTS.glob("*/manifest.json"), reverse=True):
        manifest = _read_json(root)
        found.append(
            {
                "id": root.parent.name,
                "status": manifest.get("status"),
                "created_at": manifest.get("created_at"),
            }
        )
    return found


def default_experiment() -> str | None:
    complete = [e["id"] for e in experiments() if e["status"] == "completed"]
    return complete[0] if complete else None


def _exp_root(exp: str | None) -> Path:
    exp = exp or default_experiment()
    if not exp or "/" in exp or "\\" in exp or ".." in exp:
        raise tornado.web.HTTPError(404, "no such experiment")
    root = RESULTS / exp
    if not (root / "manifest.json").exists():
        raise tornado.web.HTTPError(404, "no such experiment")
    return root


# ------------------------------------------------------------------------------------------------ live jobs


class LiveJob:
    """One live run: an append-only event list read by the SSE stream (safe to resume from any index)."""

    def __init__(self, params: dict[str, Any]) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.params = params
        self.events: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        self.done = False
        self.started = time.time()
        self.watchers = 0  # open SSE connections
        self.last_seen = time.time()
        self.cancelled = False

    def attach(self) -> None:
        with self.lock:
            self.watchers += 1
            self.last_seen = time.time()

    def detach(self) -> None:
        with self.lock:
            self.watchers -= 1
            self.last_seen = time.time()

    def abandoned(self, grace: float) -> bool:
        """True if nobody has watched this job for ``grace`` seconds (tab closed, page left, reloaded)."""
        with self.lock:
            return self.watchers <= 0 and time.time() - self.last_seen > grace

    def emit(self, event: dict[str, Any]) -> None:
        with self.lock:
            self.events.append(clean(event))

    def since(self, index: int) -> tuple[list[dict[str, Any]], bool]:
        with self.lock:
            return self.events[index:], self.done


JOBS: dict[str, LiveJob] = {}
ABANDON_AFTER_S = 6.0


class LiveCancelled(Exception):
    """Raised inside the optimizer callback to stop a live run that nobody is watching any more."""


def watchdog(job: LiveJob) -> None:
    """Cancel a live job once no browser has been connected to it for a few seconds."""
    while not job.done:
        if job.abandoned(ABANDON_AFTER_S):
            job.cancelled = True
            log.info("live job %s cancelled: no viewer for %.0f s", job.id, ABANDON_AFTER_S)
            return
        time.sleep(1.0)


class Stream(Callback):
    """Observer that forwards optimizer events to a job (it only reads; it cannot change the search)."""

    def __init__(self, job: LiveJob, method: str) -> None:
        self.job, self.method = job, method

    def on_evaluation(self, event: EvaluationEvent) -> None:
        if self.job.cancelled:
            raise LiveCancelled()
        info = event.info or {}
        self.job.emit(
            {
                "t": "eval",
                "m": self.method,
                "i": event.eval_index,
                "it": event.iteration,
                "p": event.particle_id,
                "cfg": [event.config[h] for h in HP],
                "pos": event.position,
                "vel": event.velocity,
                "f": event.fitness,
                "pbest": event.pbest_fitness,
                "gbest": event.gbest_fitness,
                "best": event.best_so_far_fitness,
                "cache": bool(info.get("cache_hit", False)),
                "cv": info.get("cv_scores"),
            }
        )

    def on_iteration_end(self, summary: IterationSummary) -> None:
        self.job.emit(
            {
                "t": "iter",
                "m": self.method,
                "it": summary.iteration,
                "gf": summary.gbest_fitness,
                "gcfg": [summary.gbest_config[h] for h in HP],
                "improved": summary.gbest_improved,
                "mean": summary.mean_fitness,
                "min": summary.min_fitness,
                "max": summary.max_fitness,
                "div": summary.diversity,
                "unique": summary.n_unique_configs,
            }
        )


def parse_params(body: dict[str, Any]) -> dict[str, Any]:
    """Validate and clamp user parameters for a live run."""
    params: dict[str, Any] = {}
    dataset = body.get("dataset", "heart_cleveland")
    if dataset not in LABELS:
        raise tornado.web.HTTPError(400, "unknown dataset")
    params["dataset"] = dataset
    defaults = {
        "n_particles": 10,
        "max_iter": 10,
        "w": 0.7298,
        "c1": 1.49618,
        "c2": 1.49618,
        "v_max_frac": 0.2,
        "fold": 0,
        "seed": None,
    }
    for key, default in defaults.items():
        value = body.get(key, default)
        if value is None:
            params[key] = None
            continue
        low, high, kind = LIMITS[key]
        try:
            params[key] = min(max(kind(value), kind(low)), kind(high))
        except (TypeError, ValueError) as exc:
            raise tornado.web.HTTPError(400, f"bad value for {key}") from exc
    if params["seed"] is None:
        params["seed"] = params["fold"]
    params["race"] = bool(body.get("race", True))
    manual = body.get("manual")
    if manual:
        params["manual"] = {
            "n_estimators": min(max(int(manual.get("n_estimators", 60)), 50), 200),
            "max_depth": min(max(int(manual.get("max_depth", 3)), 2), 20),
            "min_samples_split": min(max(int(manual.get("min_samples_split", 10)), 2), 10),
        }
    else:
        params["manual"] = None
    # the chosen configuration can seed particle 0 of the swarm and/or be scored as a comparison
    params["start_from_manual"] = bool(body.get("start_from_manual", False)) and params["manual"] is not None
    params["compare_manual"] = bool(body.get("compare_manual", True)) and params["manual"] is not None
    return params


def live_config(params: dict[str, Any]) -> ExperimentConfig:
    seeds = [0, 1, 2, 3, 4]
    seeds[params["fold"]] = params["seed"]
    overrides = [
        f"pso.n_particles={params['n_particles']}",
        f"pso.max_iter={params['max_iter']}",
        f"pso.w={params['w']}",
        f"pso.c1={params['c1']}",
        f"pso.c2={params['c2']}",
        f"pso.v_max_frac={params['v_max_frac']}",
        f"split.run_seeds=[{', '.join(map(str, seeds))}]",
    ]
    if params.get("start_from_manual") and params.get("manual"):
        m = params["manual"]
        overrides.append(f"pso.start=[{m['n_estimators']}, {m['max_depth']}, {m['min_samples_split']}]")
    return load_config([REPO / "configs" / "default.yaml"], overrides)


def _summary(record: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "best_hyperparameters",
        "best_validation_fitness",
        "n_evaluations",
        "n_unique_fits",
        "convergence_iteration",
        "stop_reason",
    )
    return {**{k: record.get(k) for k in keys}, "test": record["test_metrics"]}


def score_manual(
    cfg: ExperimentConfig, data: DatasetBundle, fold: Any, config: dict[str, int], metric: str
) -> dict[str, Any]:
    """A hand-picked configuration, scored like a candidate (inner CV), then once on the test fold."""
    settings = cfg.for_dataset(data.name)
    seed = cfg.split.run_seeds[fold.fold_index]
    evaluator = FitnessEvaluator(
        fold.opt,
        cfg.split.inner_folds,
        seed,
        metric,
        settings.preprocessing,
        n_jobs_folds=settings.fitness.n_jobs_folds,
    )
    with OptimizationPhase():
        scored = evaluator(config)
    final = final_evaluate(
        config,
        fold.opt,
        fold.test,
        seed,
        settings.preprocessing,
        labels=list(range(len(data.class_names))),
        positive_label=positive_label_for(data),
    )
    return {
        "best_hyperparameters": config,
        "best_validation_fitness": scored.fitness,
        "n_evaluations": 1,
        "n_unique_fits": 1,
        "convergence_iteration": None,
        "stop_reason": None,
        "test": final.metrics,
    }


def run_job(job: LiveJob) -> None:
    """Worker: the closed loop (and the open-loop race) on one outer fold, then the vault opens."""
    p = job.params
    try:
        cfg = live_config(p)
        data = bundle(p["dataset"])
        settings = cfg.for_dataset(data.name)
        metric = audit(data, settings.fitness.class_ratio_gate, settings.fitness.metric)["fitness_metric"]
        fold = outer_folds(data, cfg.split.outer_folds, cfg.split.outer_seed)[p["fold"]]
        budget = p["n_particles"] * (p["max_iter"] + 1)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        out = RESULTS / "live" / f"{stamp}_{data.name}_fold{p['fold']}_web_{job.id}"
        exp_id = f"web-{stamp}"
        job.emit(
            {
                "t": "start",
                "params": p,
                "budget": budget,
                "metric": metric,
                "n_opt": len(fold.opt.y),
                "n_test": len(fold.test),
                "classes": data.class_names,
                "out": out.relative_to(REPO).as_posix(),
            }
        )
        records: dict[str, dict[str, Any]] = {}
        errors: list[str] = []

        def search(method: str) -> None:
            try:
                records[method] = run_fold(
                    cfg, data, fold, method, metric, out / method, exp_id, callbacks=[Stream(job, method)]
                )
                job.emit(
                    {
                        "t": "search_done",
                        "m": method,
                        "best": records[method]["best_hyperparameters"],
                        "val": records[method]["best_validation_fitness"],
                    }
                )
            except LiveCancelled:
                return
            except Exception as exc:  # reported to the browser; the job ends cleanly
                log.exception("live %s failed", method)
                errors.append(f"{method}: {type(exc).__name__}: {exc}")

        threads = [
            threading.Thread(target=search, args=(m,), daemon=True)
            for m in (["pso", "random_search"] if p["race"] else ["pso"])
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if job.cancelled:
            job.emit({"t": "cancelled"})
            return
        if errors:
            raise RuntimeError("; ".join(errors))
        job.emit({"t": "phase", "msg": "searches finished; scoring the default RF and opening the test fold"})
        records["baseline"] = run_fold(cfg, data, fold, "baseline", metric, out / "baseline", exp_id)
        results = {m: _summary(r) for m, r in records.items()}
        if p["manual"] and p["compare_manual"]:
            results["manual"] = score_manual(cfg, data, fold, p["manual"], metric)
        job.emit(
            {"t": "vault", "results": results, "n_test": len(fold.test), "elapsed": time.time() - job.started}
        )
    except Exception as exc:
        log.exception("live job failed")
        job.emit({"t": "error", "msg": f"{type(exc).__name__}: {exc}"})
    finally:
        with job.lock:
            job.done = True


# ------------------------------------------------------------------------------------------------ handlers


class Api(tornado.web.RequestHandler):
    def set_default_headers(self) -> None:
        self.set_header("Content-Type", "application/json; charset=utf-8")
        self.set_header("Cache-Control", "no-store")

    def send(self, payload: Any) -> None:
        self.write(json.dumps(clean(payload), allow_nan=False))

    def write_error(self, status_code: int, **kwargs: Any) -> None:
        reason = self._reason
        exc = kwargs.get("exc_info")
        if exc and isinstance(exc[1], tornado.web.HTTPError) and exc[1].log_message:
            reason = exc[1].log_message
        self.write(json.dumps({"error": reason}))


class MetaHandler(Api):
    def get(self) -> None:
        cfg = load_config([REPO / "configs" / "default.yaml"])
        datasets = []
        for name in LABELS:
            data = bundle(name)
            facts = data.meta.get("audit", {})
            fold = outer_folds(data, cfg.split.outer_folds, cfg.split.outer_seed)[0]
            datasets.append(
                {
                    "key": name,
                    "label": LABELS[name],
                    "task": TASKS[name],
                    "n_samples": facts.get("n_samples"),
                    "n_features": facts.get("n_features"),
                    "n_classes": len(data.class_names),
                    "classes": data.class_names,
                    "n_opt": len(fold.opt.y),
                    "n_test": len(fold.test),
                    "landscape": (RESULTS / "landscape" / f"{name}_fold0.json").exists(),
                }
            )
        self.send(
            {
                "datasets": datasets,
                "space": {n: [b.low, b.high] for n, b in cfg.search_space.params.items()},
                "defaults": {
                    "n_particles": cfg.pso.n_particles,
                    "max_iter": cfg.pso.max_iter,
                    "w": cfg.pso.w,
                    "c1": cfg.pso.c1,
                    "c2": cfg.pso.c2,
                    "v_max_frac": cfg.pso.v_max_frac,
                },
                "limits": {k: [lo, hi] for k, (lo, hi, _) in LIMITS.items()},
                "experiments": experiments(),
                "default_experiment": default_experiment(),
            }
        )


@lru_cache(maxsize=8)
def _verify(root: str, finished_at: str | None) -> list[str]:
    return verify_results(Path(root))


class ResultsHandler(Api):
    def get(self) -> None:
        root = _exp_root(self.get_argument("exp", None))
        manifest = _read_json(root / "manifest.json")
        config = _read_json(root / "config.resolved.json")
        summary = pd.read_csv(root / "summary.csv") if (root / "summary.csv").exists() else pd.DataFrame()
        folds = (
            pd.read_csv(root / "summary_folds.csv")
            if (root / "summary_folds.csv").exists()
            else pd.DataFrame()
        )
        per_dataset = {}
        for name in config["experiment"]["datasets"]:
            if not (root / name).is_dir():
                continue
            anytime = {}
            for method in ("pso", "random_search"):
                paths = sorted(root.glob(f"{name}/fold_*/{method}/evaluations.csv"))
                if not paths:
                    continue
                ev = pd.concat(
                    [
                        pd.read_csv(p, usecols=["outer_fold", "eval_index", "best_so_far_fitness"])
                        for p in paths
                    ]
                )
                grid = ev.pivot_table(index="eval_index", columns="outer_fold", values="best_so_far_fitness")
                anytime[method] = {
                    "mean": grid.mean(axis=1).tolist(),
                    "min": grid.min(axis=1).tolist(),
                    "max": grid.max(axis=1).tolist(),
                }
            its = [pd.read_csv(p) for p in sorted(root.glob(f"{name}/fold_*/pso/iterations.csv"))]
            convergence = [
                {
                    "fold": int(i.outer_fold.iloc[0]),
                    "gbest": i.gbest_fitness.tolist(),
                    "mean": i.mean_fitness.tolist(),
                    "div": i.diversity.tolist(),
                }
                for i in its
            ]
            chosen = []
            for path in sorted(root.glob(f"{name}/fold_*/*/final.json")):
                f = _read_json(path)
                chosen.append(
                    {
                        "fold": f["outer_fold"],
                        "method": f["method"],
                        "cfg": f["best_hyperparameters"],
                        "val": f["best_validation_fitness"],
                        "test": f["test_metrics"]["accuracy"],
                        "ties": f.get("n_ties_with_best"),
                        "bounds": f.get("boundary_hits"),
                        "conv": f.get("convergence_iteration"),
                        "cm": f["test_metrics"]["confusion_matrix"],
                    }
                )
            dep = root / name / "deployment" / "pso" / "deployment.json"
            per_dataset[name] = {
                "label": LABELS.get(name, name),
                "audit": _read_json(root / name / "audit.json"),
                "anytime": anytime,
                "convergence": convergence,
                "chosen": chosen,
                "deployment": _read_json(dep) if dep.exists() else None,
            }
        self.send(
            {
                "id": root.name,
                "manifest": manifest,
                "problems": _verify(str(root), manifest.get("finished_at")),
                "summary": summary.to_dict("records"),
                "folds": folds.to_dict("records"),
                "datasets": per_dataset,
            }
        )


class ReplayHandler(Api):
    def get(self) -> None:
        root = _exp_root(self.get_argument("exp", None))
        name = self.get_argument("dataset")
        run = self.get_argument("fold", "0")
        if name not in LABELS:
            raise tornado.web.HTTPError(400, "unknown dataset")
        base = root / name / ("deployment" if run == "deployment" else f"fold_{int(run)}")
        pso = base / "pso"
        if not (pso / "evaluations.csv").exists():
            raise tornado.web.HTTPError(404, "no PSO trace for this run")
        ev = pd.read_csv(pso / "evaluations.csv")
        events = [
            {
                "i": int(r.eval_index),
                "it": int(r.iteration),
                "p": int(r.particle_id),
                "cfg": [int(r.n_estimators), int(r.max_depth), int(r.min_samples_split)],
                "pos": [r.pos_n_estimators, r.pos_max_depth, r.pos_min_samples_split],
                "vel": [r.vel_n_estimators, r.vel_max_depth, r.vel_min_samples_split],
                "f": r.fitness,
                "pbest": r.pbest_fitness,
                "gbest": r.gbest_fitness,
                "cache": bool(r.cache_hit),
            }
            for r in ev.itertuples()
        ]
        it = pd.read_csv(pso / "iterations.csv")
        iterations = [
            {
                "it": int(r.iteration),
                "gf": r.gbest_fitness,
                "gcfg": [int(r.gbest_n_estimators), int(r.gbest_max_depth), int(r.gbest_min_samples_split)],
                "improved": bool(r.gbest_improved),
                "mean": r.mean_fitness,
                "min": r.min_fitness,
                "max": r.max_fitness,
                "div": r.diversity,
            }
            for r in it.itertuples()
        ]
        rs = []
        if (base / "random_search" / "evaluations.csv").exists():
            r = pd.read_csv(base / "random_search" / "evaluations.csv")
            rs = [
                {"cfg": [int(a), int(b), int(c)], "f": f, "best": bsf}
                for a, b, c, f, bsf in zip(
                    r.n_estimators,
                    r.max_depth,
                    r.min_samples_split,
                    r.fitness,
                    r.best_so_far_fitness,
                    strict=True,
                )
            ]
        finals = {
            m: _read_json(base / m / "final.json")
            for m in ("baseline", "random_search", "pso")
            if (base / m / "final.json").exists()
        }
        self.send(
            {
                "events": events,
                "iterations": iterations,
                "random_search": rs,
                "finals": {
                    m: {
                        "cfg": f["best_hyperparameters"],
                        "val": f["best_validation_fitness"],
                        "test": f["test_metrics"]["accuracy"],
                    }
                    for m, f in finals.items()
                },
            }
        )


class LandscapeHandler(Api):
    def get(self) -> None:
        name = self.get_argument("dataset")
        path = RESULTS / "landscape" / f"{name}_fold0.json"
        if name not in LABELS or not path.exists():
            raise tornado.web.HTTPError(
                404, "landscape not measured yet (python scripts/measure_landscape.py)"
            )
        self.send(_read_json(path))


class IsolationProofHandler(Api):
    def post(self) -> None:
        test = HeldOutTestSet(np.zeros((3, 2)), np.array([0, 1, 0]), np.array([0, 1, 2]))
        try:
            with OptimizationPhase():
                test.reveal()
            outcome = {"blocked": False}
        except TestSetAccessError as exc:
            outcome = {"blocked": True, "error": str(exc)}
        X, _ = test.reveal()
        self.send({**outcome, "rows_after_phase": len(X)})


class LiveStartHandler(Api):
    def post(self) -> None:
        active = [j for j in JOBS.values() if not j.done]
        if len(active) >= MAX_ACTIVE_JOBS:
            raise tornado.web.HTTPError(429, "two live runs are already in progress; wait for one to finish")
        try:
            body = json.loads(self.request.body or b"{}")
        except json.JSONDecodeError as exc:
            raise tornado.web.HTTPError(400, "invalid JSON") from exc
        job = LiveJob(parse_params(body))
        JOBS[job.id] = job
        threading.Thread(target=run_job, args=(job,), daemon=True).start()
        threading.Thread(target=watchdog, args=(job,), daemon=True).start()
        self.send({"id": job.id, "params": job.params})


class LiveStreamHandler(tornado.web.RequestHandler):
    closed = False

    def on_connection_close(self) -> None:
        self.closed = True

    async def get(self, job_id: str) -> None:
        self.set_header("Content-Type", "text/event-stream")
        self.set_header("Cache-Control", "no-cache")
        job = JOBS.get(job_id)
        if job is None:  # e.g. the server was restarted: tell the page to stop reconnecting
            self.write("event: gone\ndata: {}\n\n")
            return
        index = int(self.get_argument("from", "0"))
        job.attach()
        try:
            while not self.closed:
                events, done = job.since(index)
                for event in events:
                    self.write(f"data: {json.dumps(event, allow_nan=False)}\n\n")
                index += len(events)
                if events:
                    await self.flush()
                if done and index >= len(job.events):
                    self.write("event: end\ndata: {}\n\n")
                    await self.flush()
                    return
                await asyncio.sleep(0.04)
        except StreamClosedError:
            return
        finally:
            job.detach()


class IndexHandler(tornado.web.RequestHandler):
    def get(self) -> None:
        self.set_header("Cache-Control", "no-store")
        self.write((STATIC / "index.html").read_bytes())


class NoCacheStatic(tornado.web.StaticFileHandler):
    def set_extra_headers(self, path: str) -> None:
        self.set_header("Cache-Control", "no-store")


def make_app() -> tornado.web.Application:
    return tornado.web.Application(
        [
            (r"/", IndexHandler),
            (r"/api/meta", MetaHandler),
            (r"/api/results", ResultsHandler),
            (r"/api/replay", ReplayHandler),
            (r"/api/landscape", LandscapeHandler),
            (r"/api/proof/isolation", IsolationProofHandler),
            (r"/api/live", LiveStartHandler),
            (r"/api/live/([0-9a-f]+)/events", LiveStreamHandler),
            (r"/static/(.*)", NoCacheStatic, {"path": str(STATIC)}),
        ]
    )


def serve(port: int = 8600, open_browser: bool = True) -> None:
    """Run the web frontend until interrupted."""
    import webbrowser

    from pso_rf.utils.log import setup_logging

    setup_logging()
    url = f"http://localhost:{port}/"

    async def main() -> None:
        make_app().listen(port, address="127.0.0.1")
        log.info("PSO × Random Forest frontend: %s  (Ctrl+C to stop)", url)
        if open_browser:
            threading.Timer(0.8, lambda: webbrowser.open(url)).start()
        await asyncio.Event().wait()

    asyncio.run(main())
