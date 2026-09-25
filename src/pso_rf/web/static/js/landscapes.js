// Objective functions for the playground over (n_estimators ∈ [50, 200], max_depth ∈ [2, 20]).
// "measured" landscapes are the real inner-CV accuracy F_0 on a grid (results/landscape/*.json, never test data).

export const LOWER = [50, 2], UPPER = [200, 20];

/** Build a lookup objective from a measured grid slice (decoded to the nearest measured grid point). */
export function measured(land, split) {
  const table = land.fitness[String(split)], ns = land.n_estimators, ds = land.max_depth;
  const n0 = ns[0], dn = ns[1] - ns[0];
  const f = (x) => {
    const ni = Math.max(0, Math.min(ns.length - 1, Math.round((x[0] - n0) / dn)));
    const di = Math.max(0, Math.min(ds.length - 1, Math.round(x[1]) - ds[0]));
    return table[di][ni];
  };
  let best = -Infinity, arg = null, lo = Infinity;
  table.forEach((row, di) => row.forEach((v, ni) => { if (v > best) { best = v; arg = [ns[ni], ds[di]]; } lo = Math.min(lo, v); }));
  const ties = table.flat().filter((v) => v === best).length;
  return { f, best, arg, lo, ties, grid: { ns, ds, table }, discrete: true,
    decode: (x) => [ns[Math.max(0, Math.min(ns.length - 1, Math.round((x[0] - n0) / dn)))], Math.max(2, Math.min(20, Math.round(x[1])))] };
}

function analytic(fn, argU) {
  // fn maps unit-square coords (u, v) to a value in [0, 1] (1 = best)
  const f = (x) => fn((x[0] - LOWER[0]) / (UPPER[0] - LOWER[0]), (x[1] - LOWER[1]) / (UPPER[1] - LOWER[1]));
  const arg = [LOWER[0] + argU[0] * (UPPER[0] - LOWER[0]), LOWER[1] + argU[1] * (UPPER[1] - LOWER[1])];
  let lo = Infinity;
  for (let i = 0; i <= 60; i++) for (let j = 0; j <= 60; j++) lo = Math.min(lo, fn(i / 60, j / 60));
  return { f, best: fn(...argU), arg, lo, ties: 1, discrete: false, decode: (x) => [Math.round(x[0]), Math.round(x[1])] };
}

const OPT = [0.68, 0.33];
export const ANALYTIC = {
  rastrigin: { label: "Rastrigin (many local optima)", make: () => analytic((u, v) => {
    const z = [(u - OPT[0]) * 7, (v - OPT[1]) * 7], raw = 20 + z.reduce((s, q) => s + q * q - 10 * Math.cos(2 * Math.PI * q), 0);
    return 1 - raw / 110; }, OPT) },
  ackley: { label: "Ackley (one deep funnel)", make: () => analytic((u, v) => {
    const z = [(u - OPT[0]) * 6, (v - OPT[1]) * 6], r = Math.sqrt((z[0] ** 2 + z[1] ** 2) / 2);
    const raw = -20 * Math.exp(-0.2 * r) - Math.exp((Math.cos(2 * Math.PI * z[0]) + Math.cos(2 * Math.PI * z[1])) / 2) + 20 + Math.E;
    return 1 - raw / 14; }, OPT) },
  plateau: { label: "Plateaus (accuracy-like steps)", make: () => analytic((u, v) => {
    const d = Math.hypot(u - OPT[0], (v - OPT[1]) * 1.2); return Math.round((1 - d) * 14) / 14; }, OPT) },
};
