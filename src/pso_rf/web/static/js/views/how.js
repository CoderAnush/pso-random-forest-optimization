// How it works: the control-system view, the equations, the evaluation protocol, and two live proofs.
import { api } from "../api.js";
import { LineChart } from "../charts.js";
import { LoopDiagram } from "../loop.js";
import { Swarm } from "../pso.js";
import { COLORS, h, toast } from "../ui.js";

export function mount(root) {
  const loopBox = h("div");
  const loop = new LoopDiagram(loopBox);
  const demo = [[120, 8, 4], [150, 12, 3], [176, 10, 2], [180, 15, 3]];
  let k = 0, gf = 0.81;
  const timer = setInterval(() => {
    const cfg = demo[k % demo.length], f = 0.8 + ((k * 37) % 40) / 1000;
    loop.setValues({ iter: Math.floor(k / 2), particle: k % 10, cfg, f, gcfg: demo[3], gf: Math.max(gf, f) });
    gf = Math.max(gf, f);
    loop.pulse(k % 2 ? "feedback" : "forward"); k++;
  }, 1300);
  let openLoop = false;
  const flip = h("button", { type: "button", class: "btn" }, "Show the open-loop comparator");
  flip.addEventListener("click", () => { openLoop = !openLoop; loop.setOpenLoop(openLoop); flip.textContent = openLoop ? "Show the closed loop (PSO)" : "Show the open-loop comparator"; });

  const mapping = [["Plant (the system)", "Random Forest, trained and validated on the optimization data"],
    ["Controller", "Particle Swarm Optimization (from scratch)"], ["Control action", "a particle's decoded hyperparameters (n_estimators, max_depth, min_samples_split)"],
    ["Measurement / feedback", "mean 5-fold inner-CV accuracy = fitness, returned to PSO"], ["Adaptation", "pbest / gbest memory, then the velocity and position update"],
    ["One cycle / one step", "one particle evaluation / one swarm iteration"], ["Open-loop contrast", "random search: same plant and budget, no feedback"]];

  // outer / inner fold protocol animation
  const protoSvg = protocol();

  // proof 1
  const p1 = h("div", { class: "proof-out" });
  const b1 = h("button", { type: "button", class: "btn primary" }, "Try to peek at the test fold");
  b1.addEventListener("click", async () => {
    try {
      const r = await api.isolation();
      p1.replaceChildren(r.blocked
        ? h("div", {}, h("span", { class: "pill ok" }, "✓ blocked"), h("div", { class: "mono", style: { marginTop: "8px", color: COLORS.ink2 } }, `TestSetAccessError: ${r.error}`),
          h("div", { style: { marginTop: "6px" } }, `After the phase closed, the final evaluation could read it (${r.rows_after_phase} rows).`))
        : h("span", { class: "pill bad" }, "✕ the test fold was readable — this must never happen"));
    } catch (err) { toast(err.message, "error"); }
  });
  // proof 2 (JS port of the IT-02 feedback ablation)
  const chartBox = h("div");
  let chart = null;
  const b2 = h("button", { type: "button", class: "btn primary" }, "Run the feedback ablation");
  b2.addEventListener("click", () => {
    const L = [50, 2, 2], U = [200, 20, 10], opt = [140, 9, 5], rg = U.map((u, d) => u - L[d]);
    const objective = (x) => -x.reduce((s, v, d) => s + ((Math.round(v) - opt[d]) / rg[d]) ** 2, 0);
    const dist = (g) => Math.sqrt(g.reduce((s, v, d) => s + ((Math.round(v) - opt[d]) / rg[d]) ** 2, 0));
    if (!chart) chart = new LineChart(chartBox, { xLabel: "PSO iteration", yLabel: "distance of gbest to the optimum", integerX: true, height: 240, yMin: 0 });
    const styles = { true: [COLORS.pso, "true feedback"], constant: [COLORS.muted, "constant feedback"], shuffled: [COLORS.gbest, "mirrored (wrong) feedback"] };
    for (const fb of ["true", "constant", "shuffled"]) {
      const s = new Swarm({ lower: L, upper: U, N: 10, w: 0.7298, c1: 1.49618, c2: 1.49618, vmaxFrac: 0.2, seed: 0, objective, feedback: fb });
      const pts = [[0, dist(s.g)]];
      while (s.t < 20) { s.step(); pts.push([s.t, dist(s.g)]); }
      chart.set(fb, { color: styles[fb][0], points: pts, step: true, width: 2.5, label: styles[fb][1], dash: fb === "constant" ? "5 4" : null });
    }
  });

  root.append(h("div", { class: "view-head" }, h("div", {}, h("h1", {}, "How the closed loop works"),
    h("p", {}, "Where the optimization is applied, what is fed back, and why the held-out test data can never leak into the search."))),
  h("div", { class: "grid" },
    h("div", { class: "grid cols-2" },
      h("div", { class: "card" }, h("h3", {}, "The loop"), loopBox, h("div", { class: "btn-row", style: { marginTop: "8px" } }, flip)),
      h("div", { class: "card" }, h("h3", {}, "Control-system view"),
        h("table", { class: "t map-table" }, h("tbody", {}, ...mapping.map(([a, b]) => h("tr", {}, h("td", {}, a), h("td", {}, b))))),
        h("div", { class: "caption", style: { marginTop: "8px" } }, "Honest framing: a static plant with no setpoint (extremum seeking), not a real-time controller."))),
    h("div", { class: "grid cols-2" },
      h("div", { class: "card" }, h("h3", {}, "PSO update (from scratch)"),
        h("div", { class: "eq", html: `v ← clip( <span class="c">w</span>·v + <span class="c">c<sub>1</sub></span>r<sub>1</sub>(p − x) + <span class="c">c<sub>2</sub></span>r<sub>2</sub>(g − x), ±v<sub>max</sub> )<br>x ← clip(x + v, l, u), &nbsp; v<sub>d</sub> ← 0 where clipped<br>θ = clip(rint(x)) ∈ Ω, &nbsp; |Ω| = 151 · 19 · 9 = 25,821` }),
        h("div", { class: "caption" }, "w = 0.7298, c₁ = c₂ = 1.49618, v_max = 0.2 × range, N = 10, T = 20 → 210 evaluations. Particles move in continuous space; positions are rounded to integers only to evaluate them.")),
      h("div", { class: "card" }, h("h3", {}, "Evaluation protocol", h("span", { class: "sub" }, "outer 5-fold · inner 5-fold")), protoSvg,
        h("div", { class: "caption" }, "Each run seals one outer fold (red). PSO sees only the other four. Inside them, 5 inner folds give the fitness. The sealed fold is scored once, after the search."))),
    h("div", { class: "grid cols-2" },
      h("div", { class: "card" }, h("h3", {}, "Proof 1 · the test fold is sealed during optimization"),
        h("div", { class: "caption", style: { marginBottom: "10px" } }, "Asks the backend to read held-out data while an optimization phase is open, exactly as a buggy search would. The test suite backs this with a row-hash spy on every fit and predict call, and with label/feature invariance tests."), b1, p1),
      h("div", { class: "card" }, h("h3", {}, "Proof 2 · the search depends on the feedback"),
        h("div", { class: "caption", style: { marginBottom: "10px" } }, "Same seed, a stub system with a known optimum, three kinds of feedback (JS port of test IT-02). Only true feedback steers the swarm to the optimum."), b2, chartBox))));

  return () => { clearInterval(timer); chart && chart.destroy(); };
}

function protocol() {
  const NS = "http://www.w3.org/2000/svg", svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", "0 0 520 200"); svg.setAttribute("width", "100%");
  const add = (tag, a, text) => { const e = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(a)) e.setAttribute(k, v); if (text) e.textContent = text; svg.append(e); return e; };
  const cells = [];
  for (let r = 0; r < 5; r++) {
    add("text", { x: 4, y: 26 + r * 32, fill: "#898781", "font-size": 11 }, `run ${r}`);
    for (let c = 0; c < 5; c++) cells.push([r, c, add("rect", { x: 50 + c * 52, y: 12 + r * 32, width: 48, height: 24, rx: 4 })]);
  }
  add("text", { x: 330, y: 26, fill: "#c3c2b7", "font-size": 12 }, "inner 5-fold CV on the 4 blue folds:");
  const inner = [];
  for (let c = 0; c < 5; c++) inner.push(add("rect", { x: 330 + c * 36, y: 40, width: 32, height: 18, rx: 3 }));
  add("text", { x: 330, y: 84, fill: "#898781", "font-size": 11 }, "train 4/5 → validate 1/5, × 5 → fitness");
  add("rect", { x: 330, y: 110, width: 14, height: 14, rx: 3, fill: "#3987e5" }); add("text", { x: 350, y: 121, fill: "#c3c2b7", "font-size": 11 }, "optimization data");
  add("rect", { x: 330, y: 134, width: 14, height: 14, rx: 3, fill: "#e66767" }); add("text", { x: 350, y: 145, fill: "#c3c2b7", "font-size": 11 }, "sealed test fold (scored once)");
  add("rect", { x: 330, y: 158, width: 14, height: 14, rx: 3, fill: "#86b6ef" }); add("text", { x: 350, y: 169, fill: "#c3c2b7", "font-size": 11 }, "inner validation fold");
  let t = 0;
  const paint = () => {
    const run = Math.floor(t / 5) % 5, iv = t % 5;
    cells.forEach(([r, c, el]) => { el.setAttribute("fill", c === r ? "#e66767" : "#3987e5"); el.setAttribute("opacity", r === run ? 1 : 0.28); });
    inner.forEach((el, c) => el.setAttribute("fill", c === iv ? "#86b6ef" : "#256abf"));
    t++;
  };
  paint(); const id = setInterval(() => { if (!svg.isConnected && t > 2) return clearInterval(id); paint(); }, 700);
  return svg;
}
