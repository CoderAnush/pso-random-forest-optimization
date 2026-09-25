// Lightweight SVG charts: live line chart (series + bands + step lines + hover crosshair) and bar/heatmap helpers.
import { h, COLORS } from "./ui.js";

const NS = "http://www.w3.org/2000/svg";
const s = (tag, attrs = {}) => {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null) el.setAttribute(k, v);
  return el;
};

function niceTicks(lo, hi, count = 5) {
  if (!(hi > lo)) { hi = lo + 1; }
  const raw = (hi - lo) / count, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((st) => raw <= st) || mag * 10;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

export class LineChart {
  /**
   * @param {HTMLElement} container
   * @param {{height?:number, xLabel?:string, yLabel?:string, yFormat?:(v:number)=>string, xFormat?:(v:number)=>string,
   *          xMin?:number, xMax?:number, yMin?:number, yMax?:number, integerX?:boolean}} opts
   */
  constructor(container, opts = {}) {
    this.o = { height: 230, yFormat: (v) => v.toFixed(3), xFormat: (v) => String(Math.round(v)), ...opts };
    this.series = new Map();
    this.markers = [];
    this.root = h("div", { class: "chart" });
    this.svg = s("svg", { role: "img", "aria-label": opts.ariaLabel || opts.yLabel || "chart" });
    this.tip = h("div", { class: "tooltip" });
    this.legendEl = h("div", { class: "legend" });
    this.root.append(this.svg, this.tip);
    container.append(this.root, this.legendEl);
    this.width = 600;
    this.ro = new ResizeObserver(() => { this.width = this.root.clientWidth || 600; this.render(); });
    this.ro.observe(this.root);
    this.svg.addEventListener("mousemove", (e) => this.hover(e));
    this.svg.addEventListener("mouseleave", () => { this.tip.style.display = "none"; this.cursor && this.cursor.remove(); });
  }
  /** points: [[x, y], …]; band: [[x, lo, hi], …]; step: draw as a step line (for best-so-far). */
  set(id, spec) { this.series.set(id, { width: 2, ...this.series.get(id), ...spec }); this.schedule(); return this; }
  append(id, point) { const sr = this.series.get(id); if (sr) { sr.points.push(point); this.schedule(); } }
  remove(id) { this.series.delete(id); this.schedule(); }
  clearAll() { this.series.clear(); this.markers = []; this.schedule(); }
  marker(x, label) { this.markers = [{ x, label }]; this.schedule(); }
  schedule() { if (!this.pending) { this.pending = true; requestAnimationFrame(() => { this.pending = false; this.render(); }); } }
  domain() {
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const sr of this.series.values()) {
      for (const [x, y] of sr.points || []) { if (y == null) continue; x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
      for (const [x, lo, hi] of sr.band || []) { if (lo == null) continue; x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, lo); y1 = Math.max(y1, hi); }
    }
    if (!Number.isFinite(x0)) { x0 = 0; x1 = 1; y0 = 0; y1 = 1; }
    if (this.o.xMin != null) x0 = this.o.xMin;
    if (this.o.xMax != null) x1 = Math.max(this.o.xMax, x1);
    if (x1 === x0) x1 = x0 + 1;
    const pad = (y1 - y0) * 0.12 || 0.01;
    y0 = this.o.yMin ?? y0 - pad; y1 = this.o.yMax ?? y1 + pad;
    return { x0, x1, y0, y1 };
  }
  render() {
    const W = this.width, H = this.o.height, m = { l: 52, r: 14, t: 10, b: this.o.xLabel ? 38 : 24 };
    const { x0, x1, y0, y1 } = (this.d = this.domain());
    const X = (x) => m.l + ((x - x0) / (x1 - x0)) * (W - m.l - m.r);
    const Y = (y) => H - m.b - ((y - y0) / (y1 - y0)) * (H - m.t - m.b);
    Object.assign(this, { X, Y, m, W, H });
    const svg = this.svg;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("height", H);
    svg.replaceChildren();
    const hasData = [...this.series.values()].some((sr) => sr.points?.length || sr.band?.length);
    for (const t of hasData ? niceTicks(y0, y1, 4) : []) {
      svg.append(s("line", { class: "gridline", x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t) }));
      const tx = s("text", { class: "tick", x: m.l - 8, y: Y(t) + 3.5, "text-anchor": "end" }); tx.textContent = this.o.yFormat(t); svg.append(tx);
    }
    const xt = !hasData ? [] : this.o.integerX ? niceTicks(x0, x1, 6).filter((v) => Number.isInteger(v)) : niceTicks(x0, x1, 6);
    for (const t of xt) {
      const tx = s("text", { class: "tick", x: X(t), y: H - m.b + 15, "text-anchor": "middle" }); tx.textContent = this.o.xFormat(t); svg.append(tx);
    }
    svg.append(s("line", { class: "axis", x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }));
    if (this.o.xLabel) { const t = s("text", { class: "axis-title", x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle" }); t.textContent = this.o.xLabel; svg.append(t); }
    if (this.o.yLabel) { const t = s("text", { class: "axis-title", x: -(m.t + H - m.b) / 2, y: 12, transform: "rotate(-90)", "text-anchor": "middle" }); t.textContent = this.o.yLabel; svg.append(t); }
    for (const sr of this.series.values()) {
      if (sr.band && sr.band.length) {
        const top = sr.band.map(([x, , hi]) => `${X(x)},${Y(hi)}`), bot = sr.band.slice().reverse().map(([x, lo]) => `${X(x)},${Y(lo)}`);
        svg.append(s("polygon", { class: "band", points: [...top, ...bot].join(" "), fill: sr.color }));
      }
    }
    for (const sr of this.series.values()) {
      const pts = (sr.points || []).filter(([, y]) => y != null);
      if (!pts.length) continue;
      let d = "";
      pts.forEach(([x, y], i) => {
        if (i === 0) d += `M${X(x)},${Y(y)}`;
        else d += sr.step ? `H${X(x)}V${Y(y)}` : `L${X(x)},${Y(y)}`;
      });
      svg.append(s("path", { class: "series", d, stroke: sr.color, "stroke-width": sr.width, "stroke-dasharray": sr.dash, opacity: sr.opacity ?? 1 }));
      if (sr.dots) for (const [x, y] of pts) svg.append(s("circle", { class: "dot", cx: X(x), cy: Y(y), r: sr.dotR || 3.5, fill: sr.color }));
      if (sr.head) { const [x, y] = pts[pts.length - 1]; svg.append(s("circle", { cx: X(x), cy: Y(y), r: 4.5, fill: sr.color, stroke: "#fff", "stroke-width": 1.5 })); }
    }
    for (const mk of this.markers) {
      svg.append(s("line", { x1: X(mk.x), x2: X(mk.x), y1: m.t, y2: H - m.b, stroke: COLORS.ink2, "stroke-dasharray": "3 4" }));
    }
    const empty = ![...this.series.values()].some((sr) => sr.points?.length || sr.band?.length);
    if (empty) { const t = s("text", { x: (m.l + W - m.r) / 2, y: (m.t + H - m.b) / 2, "text-anchor": "middle", class: "axis-title" }); t.textContent = this.o.empty || "waiting for data"; svg.append(t); }
    this.legendEl.replaceChildren(...[...this.series.values()].filter((sr) => sr.label).map((sr) =>
      h("span", {}, h("i", { class: sr.band && !sr.points?.length ? "band" : sr.dash ? "dash" : "", style: { background: sr.color, color: sr.color } }), sr.label)));
  }
  hover(e) {
    if (!this.X) return;
    const rect = this.svg.getBoundingClientRect(), px = ((e.clientX - rect.left) / rect.width) * this.W;
    const x = this.d.x0 + ((px - this.m.l) / (this.W - this.m.l - this.m.r)) * (this.d.x1 - this.d.x0);
    const rows = [];
    let snapX = null;
    for (const sr of this.series.values()) {
      if (!sr.label || !sr.points?.length) continue;
      let best = null;
      for (const p of sr.points) if (p[1] != null && (best == null || Math.abs(p[0] - x) < Math.abs(best[0] - x))) best = p;
      if (sr.step) { const before = sr.points.filter((p) => p[0] <= x && p[1] != null); if (before.length) best = before[before.length - 1]; }
      if (best) { rows.push([sr, best]); snapX ??= best[0]; }
    }
    if (!rows.length) return;
    this.cursor && this.cursor.remove();
    this.cursor = s("line", { class: "cursor", x1: this.X(snapX), x2: this.X(snapX), y1: this.m.t, y2: this.H - this.m.b });
    this.svg.append(this.cursor);
    this.tip.replaceChildren(h("div", { style: { color: COLORS.muted, marginBottom: "3px" } }, `${this.o.xLabel || "x"}: ${this.o.xFormat(snapX)}`),
      ...rows.map(([sr, p]) => h("div", {}, h("span", { class: "sw", style: { background: sr.color } }), `${sr.label}: `, h("b", {}, this.o.yFormat(p[1])))));
    this.tip.style.display = "block";
    const tx = e.clientX - rect.left + 14, flip = tx + 220 > rect.width;
    this.tip.style.left = `${flip ? e.clientX - rect.left - 14 - this.tip.offsetWidth : tx}px`;
    this.tip.style.top = `${Math.max(0, e.clientY - rect.top - 20)}px`;
  }
  destroy() { this.ro.disconnect(); }
}

/** Vertical bars with fold dots; groups = [{label, bars:[{key,label,value,color,dots:[…],dashed}]}]. */
export function barChart(container, groups, { height = 260, yFormat = (v) => v.toFixed(3), yLabel = "", yMin } = {}) {
  const root = h("div", { class: "chart" }), tip = h("div", { class: "tooltip" });
  const svg = s("svg");
  root.append(svg, tip);
  container.append(root);
  const all = groups.flatMap((g) => g.bars.flatMap((b) => [b.value, ...(b.dots || [])])).filter((v) => v != null);
  const lo = yMin ?? Math.max(0, Math.min(...all) - (Math.max(...all) - Math.min(...all)) * 0.6 - 0.02);
  const hi = Math.min(1.0, Math.max(...all) + 0.012);
  let first = true;
  const draw = () => {
    const W = root.clientWidth || 600, H = height, m = { l: 52, r: 10, t: 22, b: 30 };
    const animate = first; first = false;
    const Y = (v) => H - m.b - ((v - lo) / (hi - lo)) * (H - m.t - m.b);
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("height", H); svg.replaceChildren();
    for (const t of niceTicks(lo, hi, 4)) {
      svg.append(s("line", { class: "gridline", x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t) }));
      const tx = s("text", { class: "tick", x: m.l - 8, y: Y(t) + 3.5, "text-anchor": "end" }); tx.textContent = yFormat(t); svg.append(tx);
    }
    if (yLabel) { const t = s("text", { class: "axis-title", x: -(m.t + H - m.b) / 2, y: 12, transform: "rotate(-90)", "text-anchor": "middle" }); t.textContent = yLabel; svg.append(t); }
    const gw = (W - m.l - m.r) / groups.length;
    groups.forEach((g, gi) => {
      const n = g.bars.length, bw = Math.min(56, (gw * 0.78) / n - 4);
      const gx = m.l + gi * gw + (gw - n * (bw + 4)) / 2;
      const lab = s("text", { class: "tick", x: m.l + gi * gw + gw / 2, y: H - m.b + 18, "text-anchor": "middle" });
      lab.textContent = g.label; lab.style.fontSize = "12px"; lab.style.fill = COLORS.ink2; svg.append(lab);
      g.bars.forEach((b, bi) => {
        const x = gx + bi * (bw + 4), y = Y(b.value), bh = H - m.b - y;
        const rect = s("rect", { x, y: animate ? H - m.b : y, width: bw, height: animate ? 0 : Math.max(0, bh), rx: 4, fill: b.dashed ? "transparent" : b.color,
          stroke: b.dashed ? b.color : null, "stroke-dasharray": b.dashed ? "4 3" : null, "stroke-width": b.dashed ? 1.5 : null });
        svg.append(rect);
        if (animate) { rect.style.transition = "all .7s cubic-bezier(.2,.8,.2,1)"; requestAnimationFrame(() => requestAnimationFrame(() => { rect.setAttribute("y", y); rect.setAttribute("height", Math.max(0, bh)); })); }
        const val = s("text", { class: "tick", x: x + bw / 2, y: Math.min(y, ...(b.dots || []).map(Y)) - 6, "text-anchor": "middle" });
        val.textContent = yFormat(b.value); val.style.fill = COLORS.ink; val.style.fontWeight = 600; svg.append(val);
        (b.dots || []).forEach((dv, di) => svg.append(s("circle", { cx: x + bw / 2 + (di - (b.dots.length - 1) / 2) * Math.min(6, bw / b.dots.length), cy: Y(dv), r: 3, fill: "#fff", opacity: 0.85 })));
        const hit = s("rect", { x, y: m.t, width: bw, height: H - m.t - m.b, fill: "transparent" });
        hit.addEventListener("mousemove", (e) => {
          const r = root.getBoundingClientRect();
          tip.replaceChildren(h("div", {}, h("span", { class: "sw", style: { background: b.color } }), h("b", {}, b.label)),
            h("div", {}, `mean ${yFormat(b.value)}`), b.dots?.length ? h("div", { style: { color: COLORS.muted } }, `folds: ${b.dots.map(yFormat).join("  ")}`) : null,
            b.note ? h("div", { style: { color: COLORS.muted } }, b.note) : null);
          tip.style.display = "block"; tip.style.left = `${e.clientX - r.left + 12}px`; tip.style.top = `${e.clientY - r.top - 10}px`;
        });
        hit.addEventListener("mouseleave", () => { tip.style.display = "none"; });
        svg.append(hit);
      });
    });
  };
  const ro = new ResizeObserver(draw); ro.observe(root);
  return { destroy: () => ro.disconnect() };
}

/** Confusion-matrix heatmap (rows = true, columns = predicted). */
export function heatmap(container, matrix, labels, { title = "" } = {}) {
  const n = labels.length, cell = n > 3 ? 30 : 84, pad = n > 3 ? 26 : 92, max = Math.max(1, ...matrix.flat());
  const W = pad + n * cell + 4, H = 8 + n * cell + 26;
  if (title) container.append(h("div", { class: "caption", style: { marginBottom: "6px", color: COLORS.ink2 } }, title));
  const svg = s("svg", { viewBox: `0 -10 ${W} ${H}`, width: W, height: H, style: "max-width:100%;height:auto" });
  matrix.forEach((row, i) => row.forEach((v, j) => {
    const a = v / max, diag = i === j;
    svg.append(s("rect", { x: pad + j * cell, y: i * cell, width: cell - 2, height: cell - 2, rx: 3,
      fill: diag ? `rgba(57,135,229,${0.15 + 0.85 * a})` : v ? `rgba(230,103,103,${0.25 + 0.75 * a})` : "#222220" }));
    if (v) { const t = s("text", { x: pad + j * cell + cell / 2 - 1, y: i * cell + cell / 2 + 4, "text-anchor": "middle", class: "tick" });
      t.textContent = v; t.style.fill = "#fff"; t.style.fontSize = n > 3 ? "10px" : "15px"; t.style.fontWeight = 600; svg.append(t); }
  }));
  labels.forEach((l, i) => {
    const r = s("text", { x: pad - 6, y: i * cell + cell / 2 + 4, "text-anchor": "end", class: "tick" }); r.textContent = n > 3 ? i : l; r.style.fill = COLORS.ink2; r.style.fontSize = "12px"; svg.append(r);
    const c = s("text", { x: pad + i * cell + cell / 2 - 1, y: n * cell + 14, "text-anchor": "middle", class: "tick" }); c.textContent = n > 3 ? i : l; c.style.fill = COLORS.ink2; c.style.fontSize = "12px"; svg.append(c);
  });
  container.append(svg);
}
