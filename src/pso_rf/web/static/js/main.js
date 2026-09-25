// Hash router: each view module exports mount(container, meta) → unmount().
import { api } from "./api.js";
import { h, toast } from "./ui.js";

const VIEWS = {
  live: () => import("./views/live.js"),
  playground: () => import("./views/playground.js"),
  results: () => import("./views/results.js"),
  replay: () => import("./views/replay.js"),
  how: () => import("./views/how.js"),
};

const root = document.getElementById("view");
let unmount = null, token = 0;

async function route() {
  const name = (location.hash.replace(/^#\//, "").split("?")[0] || "live");
  const key = VIEWS[name] ? name : "live";
  document.querySelectorAll(".nav a").forEach((a) => a.classList.toggle("active", a.dataset.view === key));
  const mine = ++token;
  if (unmount) { try { unmount(); } catch (e) { console.error(e); } unmount = null; }
  root.replaceChildren(h("div", { class: "caption", style: { padding: "40px 0" } }, "Loading…"));
  try {
    const [meta, mod] = await Promise.all([api.meta(), VIEWS[key]()]);
    if (mine !== token) return;
    root.replaceChildren();
    unmount = mod.mount(root, meta) || null;
    root.focus({ preventScroll: true });
  } catch (err) {
    console.error(err);
    root.replaceChildren(h("div", { class: "card" }, h("h3", {}, "Could not load this view"), h("div", {}, String(err.message || err))));
    toast(String(err.message || err), "error");
  }
}

window.addEventListener("hashchange", route);
route();
