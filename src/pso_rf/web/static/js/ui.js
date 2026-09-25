// Small DOM helpers, formatting, colors and reusable controls.

export const COLORS = {
  pso: "#3987e5", random_search: "#d95926", baseline: "#199e70", manual: "#c3c2b7", gbest: "#e66767",
  ink: "#ffffff", ink2: "#c3c2b7", muted: "#898781", grid: "#2c2c2a", axis: "#383835", surface: "#1a1a19",
};
export const LABEL = {
  pso: "PSO (closed loop)", random_search: "Random search (open loop)", baseline: "Default RF", manual: "Your starting pick",
};
export const HP = ["n_estimators", "max_depth", "min_samples_split"];
export const BOUNDS = [[50, 200], [2, 20], [2, 10]];

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else if (k === "html") el.innerHTML = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}
export const $ = (sel, root = document) => root.querySelector(sel);

export const fmt = {
  acc: (v, d = 4) => (v == null || Number.isNaN(v) ? "–" : Number(v).toFixed(d)),
  pct: (v, d = 1) => (v == null ? "–" : `${(v * 100).toFixed(d)}%`),
  cfg: (c) => {
    if (!c) return "–";
    const a = Array.isArray(c) ? c : HP.map((k) => c[k]);
    return `(${a.map((x) => (x == null ? "∞" : x)).join(", ")})`;
  },
  delta: (v, d = 4) => (v == null ? "–" : `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(d)}`),
  num: (v) => (v == null ? "–" : Number(v).toLocaleString()),
  secs: (s) => (s < 60 ? `${s.toFixed(0)} s` : `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`),
};

export function toast(message, kind = "info", ms = 4200) {
  const box = document.getElementById("toasts");
  const el = h("div", { class: `toast ${kind === "error" ? "err" : ""}` }, message);
  box.append(el);
  setTimeout(() => el.remove(), ms);
}

/** Range slider with a live value readout. */
export function slider({ label, min, max, step = 1, value, format = (v) => v, hint, onInput }) {
  const input = h("input", { type: "range", min, max, step, value, "aria-label": label });
  const out = h("span", { class: "val" }, format(Number(value)));
  const paint = () => {
    const f = ((input.value - min) / (max - min)) * 100;
    input.style.setProperty("--fill", `${f}%`);
    out.textContent = format(Number(input.value));
  };
  input.addEventListener("input", () => { paint(); onInput && onInput(Number(input.value)); });
  paint();
  const wrap = h("div", { class: "ctl" },
    h("div", { class: "ctl-top" }, h("label", {}, label), out), input, hint ? h("div", { class: "hint" }, hint) : null);
  Object.defineProperty(wrap, "value", {
    get: () => Number(input.value),
    set: (v) => { input.value = v; paint(); },
  });
  wrap.input = input;
  return wrap;
}

export function toggle({ label, checked = false, onChange }) {
  const input = h("input", { type: "checkbox" });
  input.checked = checked;
  input.addEventListener("change", () => onChange && onChange(input.checked));
  const wrap = h("label", { class: "switch" }, input, h("span", { class: "track" }), h("span", {}, label));
  Object.defineProperty(wrap, "checked", { get: () => input.checked, set: (v) => { input.checked = v; } });
  return wrap;
}

export function segmented(options, value, onChange) {
  const wrap = h("div", { class: "seg", role: "radiogroup" });
  let current = value;
  const buttons = options.map(([val, text]) => {
    const b = h("button", { type: "button", role: "radio", "aria-checked": String(val === value) }, text);
    b.classList.toggle("on", val === value);
    b.addEventListener("click", () => {
      current = val;
      buttons.forEach((x, i) => { x.classList.toggle("on", options[i][0] === val); x.setAttribute("aria-checked", String(options[i][0] === val)); });
      onChange && onChange(val);
    });
    return b;
  });
  wrap.append(...buttons);
  Object.defineProperty(wrap, "value", { get: () => current });
  return wrap;
}

export function kpi(k, v, s = "") {
  const el = h("div", { class: "kpi" }, h("div", { class: "k" }, k), h("div", { class: "v" }, v), h("div", { class: "s" }, s));
  el.set = (value, sub, flash = false) => {
    el.querySelector(".v").textContent = value;
    if (sub != null) el.querySelector(".s").textContent = sub;
    if (flash) { el.classList.remove("flash"); void el.offsetWidth; el.classList.add("flash"); }
  };
  return el;
}

/** Sequential single-hue ramp for fitness on the dark surface (low = deep blue, high = near white). */
const RAMP = [[0, [24, 79, 149]], [0.45, [57, 135, 229]], [0.8, [134, 182, 239]], [1, [230, 240, 252]]];
export function fitnessColor(t) {
  t = Math.min(1, Math.max(0, t));
  for (let i = 1; i < RAMP.length; i++) {
    if (t <= RAMP[i][0]) {
      const [t0, c0] = RAMP[i - 1], [t1, c1] = RAMP[i], u = (t - t0) / (t1 - t0);
      return c0.map((c, k) => Math.round(c + (c1[k] - c) * u));
    }
  }
  return RAMP[RAMP.length - 1][1];
}
export const rgb = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;

export function clear(el) { while (el.firstChild) el.firstChild.remove(); }
