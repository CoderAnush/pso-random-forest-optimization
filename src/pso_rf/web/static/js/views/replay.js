// Replay a saved PSO run of the real experiment in 3-D, iteration by iteration (files only).
import { api } from "../api.js";
import { LineChart } from "../charts.js";
import { Swarm3D } from "../swarm3d.js";
import { COLORS, LABEL, fmt, h, kpi, segmented, slider, toggle } from "../ui.js";

export function mount(root, meta) {
  const st = { dataset: "digits", run: "0", t: 0, playing: false, speed: 1.6 };
  let data = null, timer = null, disposed = false;
  const dsSeg = segmented(meta.datasets.map((d) => [d.key, d.label]), st.dataset, (v) => { st.dataset = v; load(); });
  const runSeg = segmented([...[0, 1, 2, 3, 4].map((k) => [String(k), `fold ${k}`]), ["deployment", "deployment"]], st.run, (v) => { st.run = v; load(); });
  const playBtn = h("button", { type: "button", class: "btn primary" }, "▶ Play");
  playBtn.addEventListener("click", () => play(!st.playing));
  const tSlider = slider({ label: "Iteration", min: 0, max: 20, value: 0, onInput: (v) => { play(false); go(v); } });
  const speed = slider({ label: "Speed", min: 0.5, max: 6, step: 0.1, value: st.speed, format: (v) => `${v.toFixed(1)} it/s`, onInput: (v) => { st.speed = v; if (st.playing) { play(false); play(true); } } });
  const rsT = toggle({ label: "Overlay this fold's 210 random-search samples", checked: false, onChange: (v) => swarm.setVisible({ random: v }) });

  const viewport = h("div", { class: "viewport", style: { height: "560px" } });
  const hud = h("div", { class: "hud" }), hudA = h("div", { class: "big" }), hudB = h("div", { class: "big" }); hud.append(hudA, hudB);
  viewport.append(hud, h("div", { class: "legend-3d" },
    h("span", {}, h("i", { style: { background: COLORS.pso } }), "particle (color = fitness)"), h("span", {}, h("i", { style: { background: COLORS.gbest } }), "gbest"),
    h("span", {}, h("i", { style: { background: "#86b6ef", opacity: 0.6 } }), "configs evaluated so far")));
  const swarm = new Swarm3D(viewport, { autoRotate: true });
  swarm.setVisible({ random: false });

  const kIt = kpi("iteration", "–"), kG = kpi("gbest", "–"), kCfg = kpi("gbest config", "–"), kMean = kpi("swarm mean", "–"), kCache = kpi("cache hits", "–");
  const convBox = h("div");
  const conv = new LineChart(convBox, { xLabel: "PSO iteration", yLabel: "validation accuracy", integerX: true, height: 230 });
  const table = h("div", { style: { maxHeight: "300px", overflow: "auto" } });
  const finals = h("div");

  root.append(h("div", { class: "view-head" }, h("div", {}, h("h1", {}, "Swarm replay"),
    h("p", {}, "Step through a real PSO run from the saved experiment: where every particle was, the velocity that moved it, and how the swarm contracted onto its best configuration. Drag to orbit and hover a particle for its details."))),
  h("div", { class: "grid" },
    h("div", { class: "card grid", style: { gridTemplateColumns: "1fr 1fr" } }, h("div", { class: "ctl" }, h("label", {}, "Dataset"), dsSeg), h("div", { class: "ctl" }, h("label", {}, "Run"), runSeg)),
    h("div", { class: "stage-row" }, viewport,
      h("div", { class: "grid", style: { alignContent: "start" } },
        h("div", { class: "card controls" }, h("div", { class: "btn-row" }, playBtn), tSlider, speed, rsT),
        h("div", { class: "kpis", style: { gridTemplateColumns: "1fr 1fr" } }, kIt, kG, kCfg, kMean, kCache),
        h("div", { class: "card" }, h("h3", {}, "Outcome on this fold's held-out test data"), finals))),
    h("div", { class: "grid cols-2" }, h("div", { class: "card" }, h("h3", {}, "Convergence"), convBox),
      h("div", { class: "card" }, h("h3", {}, "Particles at this iteration"), table))));

  async function load() {
    play(false);
    try { data = await api.replay(null, st.dataset, st.run); } catch (err) { finals.textContent = err.message; return; }
    if (disposed) return;
    swarm.reset();
    data.random_search.forEach((r) => swarm.addRandom(r.cfg, r.f));
    swarm.setVisible({ random: rsT.checked });
    const last = data.iterations.length - 1;
    tSlider.input.max = last; tSlider.value = 0;
    conv.clearAll();
    conv.set("band", { color: COLORS.pso, band: data.iterations.map((r) => [r.it, r.min, r.max]), label: "swarm min–max" });
    conv.set("mean", { color: COLORS.ink2, points: data.iterations.map((r) => [r.it, r.mean]), dash: "5 4", width: 1.5, label: "swarm mean" });
    conv.set("gbest", { color: COLORS.pso, points: data.iterations.map((r) => [r.it, r.gf]), step: true, dots: true, width: 2.5, label: "gbest" });
    finals.replaceChildren(Object.keys(data.finals).length ? h("table", { class: "t" }, h("tbody", {}, ...["baseline", "random_search", "pso"].filter((m) => data.finals[m]).map((m) =>
      h("tr", {}, h("td", {}, h("span", { class: "sw", style: { background: COLORS[m] } }), LABEL[m]), h("td", { class: "mono" }, fmt.cfg(data.finals[m].cfg)),
        h("td", { class: "num" }, fmt.acc(data.finals[m].test)))))) : h("div", { class: "caption" }, "The deployment run uses all data, so it has no test fold. Its expected accuracy is the outer-CV mean shown in Results."));
    st.explored = 0;
    go(0);
  }
  function go(t) {
    if (!data) return;
    st.t = t; tSlider.value = t;
    const evs = data.events.filter((e) => e.it === t), it = data.iterations[t];
    const upto = data.events.filter((e) => e.it <= t);
    if (t < st.shown) { swarm.explored.clear(); st.shown = 0; }
    upto.slice(st.shown || 0).forEach((e) => swarm.addExplored(e.cfg, e.f)); st.shown = upto.length;
    evs.forEach((e) => swarm.setParticle(e.p, e.pos, e.vel, { ...e, improved: e.f === e.pbest && t > 0 }));
    swarm.setGbest(it.gcfg);
    conv.marker(t);
    const cache = evs.filter((e) => e.cache).length;
    hudA.innerHTML = `iteration <b>${t}</b> / ${data.iterations.length - 1}`;
    hudB.innerHTML = `gbest <b>${fmt.acc(it.gf)}</b> ${fmt.cfg(it.gcfg)}`;
    kIt.set(`${t}`, it.improved ? "gbest improved ★" : "no gbest change", it.improved && t > 0);
    kG.set(fmt.acc(it.gf), "validation accuracy"); kCfg.set(fmt.cfg(it.gcfg), "(n, depth, split)");
    kMean.set(fmt.acc(it.mean), `diversity ${it.div.toFixed(2)}`); kCache.set(`${cache} / ${evs.length}`, "revisited configs, answered from cache");
    table.replaceChildren(h("table", { class: "t" }, h("thead", {}, h("tr", {}, h("th", {}, "particle"), h("th", {}, "config"), h("th", { class: "num" }, "fitness"), h("th", { class: "num" }, "pbest"), h("th", {}, ""))),
      h("tbody", {}, ...evs.map((e) => h("tr", {}, h("td", {}, e.p), h("td", { class: "mono" }, fmt.cfg(e.cfg)), h("td", { class: "num" }, fmt.acc(e.f)),
        h("td", { class: "num" }, fmt.acc(e.pbest)), h("td", { class: "caption" }, e.cache ? "cached" : e.f === it.gf ? "★ = gbest" : ""))))));
  }
  function play(on) {
    st.playing = on; playBtn.textContent = on ? "❚❚ Pause" : "▶ Play";
    clearInterval(timer);
    if (on) {
      if (data && st.t >= data.iterations.length - 1) go(0);
      timer = setInterval(() => { if (!data || st.t >= data.iterations.length - 1) return play(false); go(st.t + 1); }, 1000 / st.speed);
    }
  }
  load();
  return () => { disposed = true; clearInterval(timer); swarm.dispose(); conv.destroy(); };
}
