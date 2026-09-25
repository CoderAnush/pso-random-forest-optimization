// Swarm playground: tweak PSO live on a measured Random Forest accuracy landscape (or a classic test function).
import { api } from "../api.js";
import { LineChart } from "../charts.js";
import { ANALYTIC, LOWER, UPPER, measured } from "../landscapes.js";
import { Swarm, mulberry32 } from "../pso.js";
import { Terrain3D } from "../terrain3d.js";
import { COLORS, fitnessColor, fmt, h, kpi, rgb, segmented, slider, toast } from "../ui.js";

const PRESETS = {
  "Design values": { w: 0.7298, c1: 1.49618, c2: 1.49618, vmax: 0.2 },
  "Explorer": { w: 1.05, c1: 1.2, c2: 1.2, vmax: 0.45 },
  "Herd (social)": { w: 0.4, c1: 0.3, c2: 2.6, vmax: 0.3 },
  "Loners (cognitive)": { w: 0.6, c1: 2.6, c2: 0.2, vmax: 0.2 },
  "No inertia": { w: 0.05, c1: 1.49618, c2: 1.49618, vmax: 0.2 },
  "Baby steps": { w: 0.7298, c1: 1.49618, c2: 1.49618, vmax: 0.03 },
};

export function mount(root, meta) {
  const st = { source: "measured:heart_cleveland", split: 2, N: 10, maxIter: 40, w: 0.7298, c1: 1.49618, c2: 1.49618, vmax: 0.2,
    speed: 4, seed: 7, feedback: "true", view: "map", playing: false };
  const lands = {};
  let L = null, swarm = null, disposed = false, raf = 0, lastStep = 0, guess = null, terrain = null;
  const trails = [];

  // ---------------------------------------------------------------- controls
  const sources = [
    ...meta.datasets.filter((d) => d.landscape).map((d) => [`measured:${d.key}`, `${d.label} · measured RF accuracy`]),
    ...Object.entries(ANALYTIC).map(([k, v]) => [`analytic:${k}`, v.label]),
  ];
  const srcSel = h("select", { style: { width: "100%" } }, ...sources.map(([v, t]) => h("option", { value: v }, t)));
  srcSel.value = st.source;
  srcSel.addEventListener("change", async () => { st.source = srcSel.value; await loadLandscape(); reset(); });
  const splitSeg = segmented([[2, "2"], [5, "5"], [10, "10"]], st.split, async (v) => { st.split = v; await loadLandscape(); reset(); });
  const splitRow = h("div", { class: "ctl" }, h("label", {}, "min_samples_split slice"), splitSeg,
    h("div", { class: "hint" }, "The map shows two hyperparameters; the third is fixed to one measured slice."));
  const sN = slider({ label: "Particles N", min: 2, max: 60, value: st.N, onInput: (v) => { st.N = v; reset(); } });
  const sW = slider({ label: "Inertia w", min: 0, max: 1.3, step: 0.01, value: st.w, format: (v) => v.toFixed(2), onInput: (v) => { st.w = v; live(); } });
  const sC1 = slider({ label: "Cognitive c₁ (own best)", min: 0, max: 3, step: 0.01, value: st.c1, format: (v) => v.toFixed(2), onInput: (v) => { st.c1 = v; live(); } });
  const sC2 = slider({ label: "Social c₂ (swarm best)", min: 0, max: 3, step: 0.01, value: st.c2, format: (v) => v.toFixed(2), onInput: (v) => { st.c2 = v; live(); } });
  const sV = slider({ label: "Velocity clamp", min: 0.01, max: 1, step: 0.01, value: st.vmax, format: (v) => `${Math.round(v * 100)}% of range`, onInput: (v) => { st.vmax = v; live(); } });
  const sSpeed = slider({ label: "Speed", min: 0.5, max: 30, step: 0.5, value: st.speed, format: (v) => `${v} iterations/s`, onInput: (v) => { st.speed = v; } });
  const sT = slider({ label: "Iteration budget T", min: 5, max: 200, value: st.maxIter, onInput: (v) => { st.maxIter = v; } });
  const fbSeg = segmented([["true", "true"], ["constant", "constant"], ["shuffled", "mirrored"]], st.feedback, (v) => { st.feedback = v; reset(); });
  const presets = h("div", { class: "preset-grid" }, ...Object.entries(PRESETS).map(([name, p]) => {
    const b = h("button", { type: "button", class: "btn" }, name);
    b.addEventListener("click", () => { Object.assign(st, p); sW.value = p.w; sC1.value = p.c1; sC2.value = p.c2; sV.value = p.vmax; live(); reset(); play(true); });
    return b;
  }));
  const playBtn = h("button", { type: "button", class: "btn primary" }, "▶ Play");
  const stepBtn = h("button", { type: "button", class: "btn" }, "Step");
  const resetBtn = h("button", { type: "button", class: "btn" }, "Reset");
  const seedBtn = h("button", { type: "button", class: "btn ghost" }, "New seed");
  playBtn.addEventListener("click", () => play(!st.playing));
  stepBtn.addEventListener("click", () => { play(false); doStep(); });
  resetBtn.addEventListener("click", () => reset());
  seedBtn.addEventListener("click", () => { st.seed = Math.floor(Math.random() * 1e6); reset(); play(true); });

  const side = h("aside", { class: "card controls" },
    h("div", { class: "ctl" }, h("label", {}, "Landscape"), srcSel), splitRow,
    h("div", { class: "btn-row" }, playBtn, stepBtn, resetBtn, seedBtn),
    h("div", { class: "ctl" }, h("label", {}, "Presets"), presets),
    sN, sW, sC1, sC2, sV, sSpeed, sT,
    h("div", { class: "ctl" }, h("label", {}, "Feedback"), fbSeg,
      h("div", { class: "hint" }, "Constant: every configuration scores the same. Mirrored: each particle is scored as if it stood at the opposite point of the map. Only true feedback lets the swarm find the peak.")));

  // ---------------------------------------------------------------- stage
  const viewSeg = segmented([["map", "Map (2-D)"], ["terrain", "Terrain (3-D)"]], st.view, (v) => { st.view = v; showView(); });
  const canvas = h("canvas"), mapBox = h("div", { class: "play-canvas" }, canvas);
  const terrBox = h("div", { class: "play-canvas hidden" });
  const tip = h("div", { class: "tooltip" }); mapBox.append(tip);
  const caption = h("div", { class: "caption" });
  const center = h("div", { class: "grid" }, h("div", { style: { display: "flex", justifyContent: "space-between", alignItems: "center", gap: "10px" } },
    viewSeg, h("span", { class: "caption" }, "Click the map to place your own guess")), mapBox, terrBox, caption);

  const kIt = kpi("iteration", "0"), kG = kpi("gbest", "–"), kCfg = kpi("gbest at", "–"), kGap = kpi("gap to true best", "–"), kDiv = kpi("diversity", "–");
  const chartBox = h("div");
  const chart = new LineChart(chartBox, { xLabel: "iteration", yLabel: "fitness", integerX: true, height: 200 });
  const insight = h("div", { class: "insight" });
  const trialsBtn = h("button", { type: "button", class: "btn" }, "Run 30 seeds: PSO vs random search");
  const trialsOut = h("div", { class: "caption" });
  trialsBtn.addEventListener("click", () => trials());
  const right = h("aside", { class: "grid" },
    h("div", { class: "kpis", style: { gridTemplateColumns: "1fr 1fr" } }, kIt, kG, kCfg, kGap, kDiv),
    h("div", { class: "card" }, h("h3", {}, "Convergence"), chartBox), insight,
    h("div", { class: "card" }, h("h3", {}, "Is feedback worth it here?"), h("div", { class: "caption", style: { marginBottom: "8px" } },
      "Repeats the current settings over 30 seeds against a random search with the same number of evaluations."), trialsBtn, trialsOut));

  root.append(h("div", { class: "view-head" }, h("div", {}, h("h1", {}, "Swarm playground"),
    h("p", {}, "Play with PSO live. The default landscape is the real inner-CV accuracy of a Random Forest over (n_estimators, max_depth), measured on outer fold 0's optimization data, so the swarm is searching real hyperparameter space. This runs a JavaScript port of the exact update rule, so every slider acts instantly."))),
  h("div", { class: "play" }, side, center, right));

  // ---------------------------------------------------------------- landscape + rendering
  let heat = null;
  async function loadLandscape() {
    const [kind, key] = st.source.split(":");
    splitRow.classList.toggle("hidden", kind !== "measured");
    if (kind === "measured") {
      lands[key] ??= await api.landscape(key);
      L = measured(lands[key], st.split);
      L.name = `${meta.datasets.find((d) => d.key === key).label}, min_samples_split = ${st.split}`;
      caption.textContent = `Measured grid: n_estimators 50–200 (step 10) × max_depth 2–20 = 304 real Random Forest evaluations per slice, inner 5-fold CV on outer fold 0 (${lands[key].n_opt_samples} samples; never test data). Positions are decoded to the nearest measured grid point. ${L.ties} grid point${L.ties > 1 ? "s tie" : " holds"} the best accuracy ${fmt.acc(L.best)}.`;
    } else {
      L = ANALYTIC[key].make(); L.name = ANALYTIC[key].label;
      caption.textContent = "A classic optimization test function, mapped onto the same two axes, to show how PSO behaves on harder shapes.";
    }
    buildHeat();
    if (terrain) terrain.setLandscape(L);
  }
  function buildHeat() {
    heat = document.createElement("canvas");
    if (L.discrete) {
      const { ns, ds, table } = L.grid; heat.width = ns.length; heat.height = ds.length;
      const ctx = heat.getContext("2d"), img = ctx.createImageData(ns.length, ds.length);
      table.forEach((row, di) => row.forEach((v, ni) => {
        const c = fitnessColor((v - L.lo) / Math.max(1e-9, L.best - L.lo)), o = ((ds.length - 1 - di) * ns.length + ni) * 4;
        img.data.set([c[0], c[1], c[2], 255], o);
      }));
      ctx.putImageData(img, 0, 0);
    } else {
      heat.width = 180; heat.height = 130;
      const ctx = heat.getContext("2d"), img = ctx.createImageData(180, 130);
      for (let j = 0; j < 130; j++) for (let i = 0; i < 180; i++) {
        const x = [LOWER[0] + (i / 179) * (UPPER[0] - LOWER[0]), UPPER[1] - (j / 129) * (UPPER[1] - LOWER[1])];
        const c = fitnessColor((L.f(x) - L.lo) / Math.max(1e-9, L.best - L.lo)); img.data.set([c[0], c[1], c[2], 255], (j * 180 + i) * 4);
      }
      ctx.putImageData(img, 0, 0);
    }
  }
  const M = { l: 46, r: 12, t: 12, b: 36 };
  let geom = null;
  function toPx(x) {
    const { W, H } = geom;
    const cellN = L.discrete ? (UPPER[0] - LOWER[0]) / (L.grid.ns.length - 1) / 2 : 0, cellD = L.discrete ? 0.5 : 0;
    const x0 = LOWER[0] - cellN, x1 = UPPER[0] + cellN, y0 = LOWER[1] - cellD, y1 = UPPER[1] + cellD;
    return [M.l + ((x[0] - x0) / (x1 - x0)) * (W - M.l - M.r), H - M.b - ((x[1] - y0) / (y1 - y0)) * (H - M.t - M.b)];
  }
  function fromPx(px, py) {
    const { W, H } = geom;
    const cellN = L.discrete ? (UPPER[0] - LOWER[0]) / (L.grid.ns.length - 1) / 2 : 0, cellD = L.discrete ? 0.5 : 0;
    const x0 = LOWER[0] - cellN, x1 = UPPER[0] + cellN, y0 = LOWER[1] - cellD, y1 = UPPER[1] + cellD;
    return [x0 + ((px - M.l) / (W - M.l - M.r)) * (x1 - x0), y0 + ((H - M.b - py) / (H - M.t - M.b)) * (y1 - y0)];
  }
  function draw(disp) {
    const dpr = Math.min(2, window.devicePixelRatio), W = mapBox.clientWidth, H = mapBox.clientHeight;
    if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) { canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr); }
    geom = { W, H };
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
    ctx.imageSmoothingEnabled = !L.discrete;
    ctx.drawImage(heat, M.l, M.t, W - M.l - M.r, H - M.t - M.b);
    ctx.fillStyle = COLORS.muted; ctx.font = "11px Inter, Segoe UI, sans-serif"; ctx.textAlign = "center";
    for (const n of [50, 75, 100, 125, 150, 175, 200]) { const [px] = toPx([n, 2]); ctx.fillText(n, px, H - M.b + 15); }
    ctx.fillText("n_estimators", (M.l + W - M.r) / 2, H - 5);
    ctx.textAlign = "right";
    for (const d of [2, 5, 8, 11, 14, 17, 20]) { const [, py] = toPx([50, d]); ctx.fillText(d, M.l - 7, py + 4); }
    ctx.save(); ctx.translate(12, (M.t + H - M.b) / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = "center"; ctx.fillText("max_depth", 0, 0); ctx.restore();
    // true optimum
    const [ox, oy] = toPx(L.arg);
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5; star(ctx, ox, oy, 9, 4); ctx.stroke();
    // user guess
    if (guess) { const [gx, gy] = toPx(guess.x); ctx.setLineDash([4, 3]); ctx.strokeStyle = COLORS.manual; ctx.strokeRect(gx - 8, gy - 8, 16, 16); ctx.setLineDash([]);
      ctx.fillStyle = COLORS.manual; ctx.textAlign = "left"; ctx.fillText("you", gx + 11, gy - 8); }
    if (!swarm) return;
    // trails
    ctx.lineWidth = 1.2;
    trails.forEach((tr, i) => { if (i >= disp.length) return; ctx.beginPath(); tr.forEach((p, j) => { const [px, py] = toPx(p); j ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
      ctx.strokeStyle = "rgba(205,226,251,.28)"; ctx.stroke(); });
    // pbest ghosts + tethers
    swarm.P.forEach((p, i) => { const [px, py] = toPx(p), [qx, qy] = toPx(disp[i]);
      ctx.strokeStyle = "rgba(134,182,239,.25)"; ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(qx, qy); ctx.lineTo(px, py); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = "rgba(134,182,239,.55)"; ctx.beginPath(); ctx.arc(px, py, 2.6, 0, 7); ctx.fill(); });
    // velocity arrows
    swarm.V.forEach((v, i) => { const a = toPx(disp[i]), b = toPx([disp[i][0] + v[0], disp[i][1] + v[1]]); arrow(ctx, a, b); });
    // particles
    disp.forEach((x) => { const [px, py] = toPx(x), f = L.f(x), c = fitnessColor((f - L.lo) / Math.max(1e-9, L.best - L.lo));
      ctx.beginPath(); ctx.arc(px, py, 6, 0, 7); ctx.fillStyle = rgb(c); ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = "#fff"; ctx.stroke(); });
    // gbest
    const [gx, gy] = toPx(swarm.g), pulse = 10 + 3 * Math.sin(performance.now() / 180);
    ctx.strokeStyle = COLORS.gbest; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.arc(gx, gy, pulse, 0, 7); ctx.stroke();
    ctx.lineWidth = 1; ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.moveTo(M.l, gy); ctx.lineTo(W - M.r, gy); ctx.moveTo(gx, M.t); ctx.lineTo(gx, H - M.b); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = COLORS.gbest; ctx.textAlign = "left"; ctx.font = "600 12px Inter, Segoe UI, sans-serif"; ctx.fillText("gbest", gx + pulse + 4, gy - 4);
  }
  function star(ctx, x, y, R, r) { ctx.beginPath(); for (let i = 0; i < 10; i++) { const a = (Math.PI / 5) * i - Math.PI / 2, rad = i % 2 ? r : R; ctx.lineTo(x + rad * Math.cos(a), y + rad * Math.sin(a)); } ctx.closePath(); }
  function arrow(ctx, [ax, ay], [bx, by]) {
    const len = Math.hypot(bx - ax, by - ay); if (len < 3) return;
    ctx.strokeStyle = "rgba(255,255,255,.75)"; ctx.fillStyle = "rgba(255,255,255,.75)"; ctx.lineWidth = 1.3;
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
    const a = Math.atan2(by - ay, bx - ax); ctx.beginPath(); ctx.moveTo(bx, by);
    ctx.lineTo(bx - 6 * Math.cos(a - 0.45), by - 6 * Math.sin(a - 0.45)); ctx.lineTo(bx - 6 * Math.cos(a + 0.45), by - 6 * Math.sin(a + 0.45)); ctx.fill();
  }

  // ---------------------------------------------------------------- simulation
  function newSwarm(seed = st.seed) {
    return new Swarm({ lower: LOWER, upper: UPPER, N: st.N, w: st.w, c1: st.c1, c2: st.c2, vmaxFrac: st.vmax, seed, objective: L.f, feedback: st.feedback });
  }
  function reset() {
    if (!L) return;
    swarm = newSwarm(); trails.length = 0; swarm.X.forEach((x) => trails.push([x.slice()]));
    chart.clearAll();
    chart.set("band", { color: COLORS.pso, band: [], label: "swarm min–max" });
    chart.set("mean", { color: COLORS.ink2, points: [], dash: "5 4", width: 1.5, label: "swarm mean" });
    chart.set("gbest", { color: COLORS.pso, points: [], step: true, width: 2.5, label: "gbest (true fitness)" });
    chart.set("opt", { color: "#ffffff", points: [[0, L.best], [st.maxIter, L.best]], dash: "2 4", width: 1, label: "true best" });
    record(); updateStats();
  }
  function trueF(x) { return L.f(x); }
  function record() {
    const hst = swarm.history[swarm.history.length - 1], tf = swarm.F.map((_, i) => trueF(swarm.X[i]));
    chart.append("gbest", [swarm.t, trueF(swarm.g)]);
    chart.append("mean", [swarm.t, tf.reduce((a, b) => a + b, 0) / tf.length]);
    chart.series.get("band").band.push([swarm.t, Math.min(...tf), Math.max(...tf)]);
    chart.o.xMax = Math.max(st.maxIter, swarm.t); chart.schedule();
    return hst;
  }
  function doStep() {
    if (!swarm) return;
    swarm.step();
    swarm.X.forEach((x, i) => { trails[i] = trails[i] || []; trails[i].push(x.slice()); if (trails[i].length > 14) trails[i].shift(); });
    record(); updateStats();
    if (swarm.t >= st.maxIter) { play(false); toast(`Budget reached: ${swarm.evals} evaluations. gbest ${fmt.acc(trueF(swarm.g))} vs true best ${fmt.acc(L.best)}.`); }
  }
  function updateStats() {
    const g = trueF(swarm.g), gap = L.best - g, dec = L.decode(swarm.g);
    kIt.set(`${swarm.t}`, `${swarm.evals} evaluations`);
    kG.set(fmt.acc(g), swarm.improved && swarm.t > 0 ? "improved ★" : "", swarm.improved && swarm.t > 0);
    kCfg.set(`(${dec[0]}, ${dec[1]})`, "(n_estimators, max_depth)");
    kGap.set(gap <= 1e-12 ? "0 · found it" : fmt.acc(gap), gap <= 1e-12 ? `reached at iteration ${swarm.gIter}` : "true best − gbest");
    kDiv.set(swarm.diversity().toFixed(2), swarm.diversity() < 0.05 ? "collapsed" : swarm.diversity() < 0.2 ? "converging" : "exploring");
    insight.innerHTML = explain();
  }
  function explain() {
    if (st.feedback === "constant") return "<b>No information.</b> Every configuration scores the same, so no personal or global best ever changes after the start. The swarm just orbits its first guesses: optimization without feedback is blind.";
    if (st.feedback === "shuffled") return "<b>Wrong information.</b> Each particle is scored as if it stood at the mirror-image point. The swarm still converges confidently, but to the wrong place. Feedback steers the search; bad feedback steers it wrong.";
    if (st.w > 0.95) return "<b>High inertia.</b> Particles keep most of their momentum, overshoot and keep exploring. Great for escaping local optima, slow to settle.";
    if (st.w < 0.25) return "<b>Almost no inertia.</b> Particles forget their motion and jump straight toward their bests. The swarm collapses fast, and can freeze on a sub-optimal plateau.";
    if (st.c2 > 2 * st.c1 + 0.3) return "<b>Herd behaviour.</b> The social pull dominates, so the whole swarm rushes to gbest. Fast agreement, high risk of premature convergence.";
    if (st.c1 > 2 * st.c2 + 0.3) return "<b>Individualists.</b> Each particle mostly trusts its own best, so the swarm splits into many local searches and agrees slowly.";
    if (st.vmax < 0.06) return "<b>Tiny steps.</b> The velocity clamp limits how far a particle moves per iteration, so it cannot cross the map within the budget.";
    return "<b>Balanced (constriction-equivalent) settings:</b> w = 0.7298, c₁ = c₂ = 1.49618. Momentum, personal memory and social pull are in the standard balance that is proven to converge (Clerc & Kennedy, 2002). These are the values used in the experiment.";
  }
  function trials() {
    const budget = st.N * (st.maxIter + 1), res = { pso: [], rs: [] };
    for (let k = 0; k < 30; k++) {
      const s = newSwarm(1000 + k); while (s.t < st.maxIter) s.step();
      res.pso.push({ best: trueF(s.g), hit: L.best - trueF(s.g) <= 1e-12 });
      const r = mulberry32(5000 + k); let best = -Infinity;
      for (let e = 0; e < budget; e++) { const x = [LOWER[0] + r() * (UPPER[0] - LOWER[0]), LOWER[1] + r() * (UPPER[1] - LOWER[1])]; best = Math.max(best, L.f(x)); }
      res.rs.push({ best, hit: L.best - best <= 1e-12 });
    }
    const row = (k, label, color) => { const a = res[k], m = a.reduce((s, x) => s + x.best, 0) / a.length, hits = a.filter((x) => x.hit).length;
      return h("tr", {}, h("td", {}, h("span", { class: "sw", style: { background: color } }), label), h("td", { class: "num" }, `${hits}/30`), h("td", { class: "num" }, fmt.acc(m))); };
    trialsOut.replaceChildren(h("table", { class: "t", style: { marginTop: "10px" } },
      h("thead", {}, h("tr", {}, h("th", {}, "method"), h("th", { class: "num" }, "found true best"), h("th", { class: "num" }, "mean final"))),
      h("tbody", {}, row("pso", "PSO", COLORS.pso), row("rs", "Random search", COLORS.random_search))),
      h("div", { style: { marginTop: "6px" } }, `Budget ${budget} evaluations each, feedback: ${st.feedback}. ${L.discrete ? "Many configurations tie at the top, which is why random search often does well here, as in the real experiment." : ""}`));
  }
  function live() { if (swarm) Object.assign(swarm, { w: st.w, c1: st.c1, c2: st.c2, vmaxFrac: st.vmax }); if (swarm) insight.innerHTML = explain(); }
  function play(on) { st.playing = on; playBtn.textContent = on ? "❚❚ Pause" : "▶ Play"; if (on && swarm && swarm.t >= st.maxIter) reset(); lastStep = performance.now(); }

  function tick(now) {
    if (disposed) return;
    raf = requestAnimationFrame(tick);
    if (!swarm || !L) return;
    const interval = 1000 / st.speed;
    if (st.playing && now - lastStep >= interval) { lastStep = now; doStep(); }
    const u = st.playing ? Math.min(1, (now - lastStep) / Math.min(interval, 450)) : 1, e = 1 - (1 - u) ** 3;
    const disp = swarm.X.map((x, i) => (swarm.prevX && swarm.t > 0 ? x.map((v, d) => swarm.prevX[i][d] + (v - swarm.prevX[i][d]) * e) : x));
    if (st.view === "map") draw(disp); else if (terrain) terrain.setSwarm(disp, swarm.P, swarm.g);
  }
  function showView() {
    mapBox.classList.toggle("hidden", st.view !== "map"); terrBox.classList.toggle("hidden", st.view !== "terrain");
    if (st.view === "terrain" && !terrain) { terrain = new Terrain3D(terrBox); terrain.setLandscape(L); }
  }
  canvas.addEventListener("mousemove", (e) => {
    if (!geom || !L) return;
    const r = canvas.getBoundingClientRect(), x = fromPx(e.clientX - r.left, e.clientY - r.top);
    if (x[0] < LOWER[0] - 6 || x[0] > UPPER[0] + 6 || x[1] < LOWER[1] - 0.6 || x[1] > UPPER[1] + 0.6) { tip.style.display = "none"; return; }
    const d = L.decode(x);
    tip.replaceChildren(h("div", {}, `n_estimators ${d[0]} · max_depth ${d[1]}`), h("div", {}, "fitness ", h("b", {}, fmt.acc(L.f(x)))));
    tip.style.display = "block"; tip.style.left = `${e.clientX - r.left + 14}px`; tip.style.top = `${e.clientY - r.top + 8}px`;
  });
  canvas.addEventListener("mouseleave", () => { tip.style.display = "none"; });
  canvas.addEventListener("click", (e) => {
    const r = canvas.getBoundingClientRect(), x = fromPx(e.clientX - r.left, e.clientY - r.top);
    x[0] = Math.max(LOWER[0], Math.min(UPPER[0], x[0])); x[1] = Math.max(LOWER[1], Math.min(UPPER[1], x[1]));
    const f = L.f(x), d = L.decode(x); guess = { x, f };
    const pct = L.best > L.lo ? ((f - L.lo) / (L.best - L.lo)) * 100 : 100;
    toast(`Your guess (${d[0]}, ${d[1]}): ${fmt.acc(f)}. True best ${fmt.acc(L.best)}; you reached ${pct.toFixed(0)}% of the landscape's range.`);
  });

  (async () => { await loadLandscape(); reset(); play(true); raf = requestAnimationFrame(tick); })()
    .catch((err) => toast(err.message, "error", 8000));

  return () => { disposed = true; cancelAnimationFrame(raf); chart.destroy(); terrain && terrain.dispose(); };
}
