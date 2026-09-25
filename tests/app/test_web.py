"""Web frontend backend (ADR-027): every endpoint, input validation, and a small live job streamed over SSE."""

from __future__ import annotations

import json
import time

import pytest
from tornado.testing import AsyncHTTPTestCase

from pso_rf.web import server


class WebTest(AsyncHTTPTestCase):
    def get_app(self):
        return server.make_app()

    def json(self, path: str, **kw):
        res = self.fetch(path, **kw)
        return res.code, json.loads(res.body or b"{}")

    def test_index_and_static(self) -> None:
        assert b"Swarm Lab" in self.fetch("/").body
        for path in ("/static/js/main.js", "/static/styles.css", "/static/vendor/three.module.js"):
            assert self.fetch(path).code == 200, path

    def test_meta(self) -> None:
        code, meta = self.json("/api/meta")
        assert code == 200
        assert [d["key"] for d in meta["datasets"]] == ["iris", "digits", "heart_cleveland"]
        assert meta["space"] == {
            "n_estimators": [50, 200],
            "max_depth": [2, 20],
            "min_samples_split": [2, 10],
        }
        assert meta["default_experiment"]

    def test_results_replay_landscape(self) -> None:
        code, r = self.json("/api/results")
        assert code == 200 and len(r["summary"]) == 9 and r["problems"] == []
        heart = r["datasets"]["heart_cleveland"]
        assert set(heart["anytime"]) == {"pso", "random_search"} and len(heart["convergence"]) == 5
        code, rp = self.json("/api/replay?dataset=iris&fold=0")
        assert code == 200 and len(rp["events"]) == 210 and len(rp["iterations"]) == 21
        assert len(rp["random_search"]) == 210 and set(rp["finals"]) == {"baseline", "random_search", "pso"}
        code, dep = self.json("/api/replay?dataset=iris&fold=deployment")
        assert code == 200 and dep["finals"] == {}
        code, land = self.json("/api/landscape?dataset=heart_cleveland")
        assert code == 200 and len(land["fitness"]["2"]) == 19 and len(land["fitness"]["2"][0]) == 16

    def test_bad_inputs(self) -> None:
        assert self.fetch("/api/results?exp=../secrets").code == 404
        assert self.fetch("/api/replay?dataset=nope&fold=0").code == 400
        assert self.fetch("/api/landscape?dataset=nope").code == 404
        assert self.fetch("/api/live", method="POST", body="{not json").code == 400
        assert self.fetch("/api/live", method="POST", body=json.dumps({"dataset": "mnist"})).code == 400
        gone = self.fetch("/api/live/ffffffffffff/events")
        assert gone.code == 200 and b"event: gone" in gone.body  # the page stops reconnecting

    def test_isolation_proof(self) -> None:
        code, r = self.json("/api/proof/isolation", method="POST", body="")
        assert (
            code == 200 and r["blocked"] and "OptimizationPhase" in r["error"] and r["rows_after_phase"] == 3
        )

    @pytest.mark.slow
    def test_live_job_streams_the_closed_loop(self) -> None:
        body = {
            "dataset": "iris",
            "n_particles": 3,
            "max_iter": 1,
            "race": True,
            "manual": {"n_estimators": 60, "max_depth": 3, "min_samples_split": 10},
        }
        code, job = self.json("/api/live", method="POST", body=json.dumps(body))
        assert code == 200
        deadline = time.time() + 240
        while not server.JOBS[job["id"]].done and time.time() < deadline:
            time.sleep(0.2)
        stream = self.fetch(f"/api/live/{job['id']}/events", request_timeout=60).body.decode()
        events = [
            json.loads(line[6:])
            for line in stream.splitlines()
            if line.startswith("data: ") and line != "data: {}"
        ]
        kinds = [e["t"] for e in events]
        assert kinds[0] == "start" and kinds[-1] == "vault", kinds[-3:]
        pso = [e for e in events if e["t"] == "eval" and e["m"] == "pso"]
        rs = [e for e in events if e["t"] == "eval" and e["m"] == "random_search"]
        assert len(pso) == len(rs) == 3 * (1 + 1)  # equal budgets N × (T + 1)
        assert [e["it"] for e in events if e["t"] == "iter"] == [0, 1]
        vault = events[-1]["results"]
        assert set(vault) == {"pso", "random_search", "baseline", "manual"}
        assert all(0 <= r["test"]["accuracy"] <= 1 for r in vault.values())
        # the test metrics are released only after both searches finished
        first_vault = kinds.index("vault")
        assert all(k != "eval" for k in kinds[first_vault:])


def test_parse_params_clamps() -> None:
    p = server.parse_params(
        {"dataset": "iris", "n_particles": 999, "w": -5, "fold": 9, "manual": {"max_depth": 99}}
    )
    assert p["n_particles"] == 30 and p["w"] == 0.05 and p["fold"] == 4 and p["seed"] == 4
    assert p["manual"]["max_depth"] == 20


def test_abandoned_live_job_is_cancelled(monkeypatch) -> None:
    """A run whose browser tab went away stops, instead of competing with the next run for the CPU."""
    monkeypatch.setattr(server, "ABANDON_AFTER_S", 0.5)
    job = server.LiveJob(
        server.parse_params({"dataset": "digits", "n_particles": 10, "max_iter": 20, "race": True})
    )
    runner = __import__("threading").Thread(target=server.run_job, args=(job,))
    runner.start()
    server.watchdog(job)  # nobody attached: cancels after the grace period
    runner.join(120)
    assert not runner.is_alive() and job.cancelled
    kinds = [e["t"] for e in job.events]
    assert kinds[-1] == "cancelled" and "vault" not in kinds
    assert sum(e["t"] == "eval" for e in job.events) < 2 * 210  # stopped well before the full budget
