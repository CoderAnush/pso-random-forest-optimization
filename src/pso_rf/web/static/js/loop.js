// Animated closed-loop diagram: packets travel the forward path on each evaluation and the feedback edge on each update.
const NS = "http://www.w3.org/2000/svg";
const s = (tag, attrs = {}, text) => {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (text != null) el.textContent = text;
  return el;
};

const NODES = {
  pso: { x: 16, y: 18, k: "① controller", t: "PSO swarm" },
  hp: { x: 208, y: 18, k: "② control action", t: "Hyperparameters" },
  rf: { x: 400, y: 18, k: "③ plant", t: "Random Forest" },
  fit: { x: 400, y: 150, k: "④ measurement", t: "Fitness (inner CV)" },
  upd: { x: 16, y: 150, k: "⑤ adaptation", t: "pbest / gbest update" },
};
const W = 172, H = 92;

export class LoopDiagram {
  constructor(container, { openLoop = false } = {}) {
    this.svg = s("svg", { class: "loop-svg", viewBox: "0 0 590 272", role: "img",
      "aria-label": "Closed loop: PSO proposes hyperparameters, the Random Forest is trained, validation fitness is fed back to PSO" });
    const defs = s("defs");
    const marker = s("marker", { id: "arr", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" });
    marker.append(s("path", { d: "M0,0 L10,5 L0,10 z", fill: "#5a5f66" }));
    defs.append(marker); this.svg.append(defs);
    const mid = (a, side) => {
      const n = NODES[a];
      return { r: [n.x + W, n.y + H / 2], l: [n.x, n.y + H / 2], b: [n.x + W / 2, n.y + H], t: [n.x + W / 2, n.y] }[side];
    };
    const path = (d, cls) => { const p = s("path", { d, class: `edge ${cls || ""}`, "marker-end": "url(#arr)" }); this.svg.append(p); return p; };
    const [a1, b1] = [mid("pso", "r"), mid("hp", "l")], [a2, b2] = [mid("hp", "r"), mid("rf", "l")];
    const [a3, b3] = [mid("rf", "b"), mid("fit", "t")], [a4, b4] = [mid("fit", "l"), mid("upd", "r")], [a5, b5] = [mid("upd", "t"), mid("pso", "b")];
    this.edges = {
      e1: path(`M${a1[0]},${a1[1]} L${b1[0] - 3},${b1[1]}`), e2: path(`M${a2[0]},${a2[1]} L${b2[0] - 3},${b2[1]}`),
      e3: path(`M${a3[0]},${a3[1]} L${b3[0]},${b3[1] - 3}`),
      e4: path(`M${a4[0]},${a4[1]} L${b4[0] + 3},${b4[1]}`, "feedback"), e5: path(`M${a5[0]},${a5[1]} L${b5[0]},${b5[1] + 3}`, "feedback"),
    };
    const fl = s("text", { class: "edge-label", x: 294, y: 264, "text-anchor": "middle" }, "feedback: fitness → memory → new velocities → new positions");
    this.svg.append(fl);
    this.feedbackLabel = fl;
    this.cut = s("g", { opacity: 0 });
    this.cut.append(s("circle", { cx: 294, cy: 196, r: 13, fill: "#1a1a19", stroke: "#e66767", "stroke-width": 1.5 }), s("text", { x: 294, y: 201, "text-anchor": "middle", fill: "#e66767", "font-size": 15 }, "✕"));
    this.svg.append(this.cut);
    this.nodes = {}; this.vals = {};
    for (const [key, n] of Object.entries(NODES)) {
      const g = s("g", { class: "node" });
      g.append(s("rect", { x: n.x, y: n.y, width: W, height: H, rx: 12 }), s("text", { class: "k", x: n.x + 12, y: n.y + 20 }, n.k),
        s("text", { class: "t", x: n.x + 12, y: n.y + 40 }, n.t));
      const v1 = s("text", { class: "v", x: n.x + 12, y: n.y + 60 }, "–"), v2 = s("text", { class: "v", x: n.x + 12, y: n.y + 78 }, "");
      g.append(v1, v2); this.svg.append(g);
      this.nodes[key] = g; this.vals[key] = [v1, v2]; this.nodes[key].title = g.querySelector(".t");
    }
    this.packet = s("circle", { class: "packet", r: 5.5, opacity: 0 });
    this.svg.append(this.packet);
    container.append(this.svg);
    this.setOpenLoop(openLoop);
    this.setValues({});
    this.busy = false;
  }
  setOpenLoop(open) {
    this.open = open;
    this.nodes.pso.title.textContent = open ? "Random sampler" : "PSO swarm";
    this.nodes.upd.title.textContent = open ? "(no update)" : "pbest / gbest update";
    this.edges.e4.classList.toggle("cut", open); this.edges.e5.classList.toggle("cut", open);
    this.cut.setAttribute("opacity", open ? 1 : 0);
    this.feedbackLabel.textContent = open ? "open loop: the next sample ignores every fitness value" : "feedback: fitness → memory → new velocities → new positions";
  }
  setValues({ iter, total, particle, cfg, f, gcfg, gf, folds }) {
    const set = (k, a, b) => { this.vals[k][0].textContent = a; this.vals[k][1].textContent = b; };
    set("pso", iter == null ? "idle" : `iteration ${iter}${total != null ? ` / ${total}` : ""}`, particle == null ? "" : `particle ${particle}`);
    set("hp", cfg ? `(${cfg.join(", ")})` : "(n, depth, split)", "= clip(rint(x))");
    set("rf", folds ? `${folds} forests on 4/5` : "trained on 4/5", "validated on 1/5");
    set("fit", f == null ? "–" : `accuracy ${Number(f).toFixed(4)}`, "mean of 5 inner folds");
    set("upd", gf == null ? "gbest –" : `gbest ${Number(gf).toFixed(4)}`, gcfg ? `(${gcfg.join(", ")})` : "");
  }
  light(keys) { for (const [k, g] of Object.entries(this.nodes)) g.classList.toggle("on", keys.includes(k)); }
  /** Animate a packet along a sequence of edges. */
  pulse(kind) {
    if (this.busy || document.hidden) return;
    const edges = kind === "feedback" ? ["e4", "e5"] : ["e1", "e2", "e3"];
    const lights = kind === "feedback" ? ["fit", "upd", "pso"] : ["pso", "hp", "rf", "fit"];
    this.busy = true; this.light(lights);
    for (const e of edges) this.edges[e].classList.add("on");
    const paths = edges.map((e) => this.edges[e]), lengths = paths.map((p) => p.getTotalLength()), total = lengths.reduce((a, b) => a + b, 0);
    const dur = kind === "feedback" ? 520 : 420, t0 = performance.now();
    this.packet.style.fill = kind === "feedback" ? "#86b6ef" : "#fff";
    const step = (now) => {
      const u = Math.min(1, (now - t0) / dur);
      let d = u * total, i = 0;
      while (i < paths.length - 1 && d > lengths[i]) { d -= lengths[i]; i++; }
      const pt = paths[i].getPointAtLength(Math.min(d, lengths[i]));
      this.packet.setAttribute("cx", pt.x); this.packet.setAttribute("cy", pt.y); this.packet.setAttribute("opacity", 1);
      if (u < 1) requestAnimationFrame(step);
      else { this.packet.setAttribute("opacity", 0); for (const e of edges) this.edges[e].classList.remove("on"); this.busy = false; }
    };
    requestAnimationFrame(step);
  }
}
