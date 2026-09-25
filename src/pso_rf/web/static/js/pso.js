// A JavaScript port of the project's PSO update (MATHEMATICAL_FORMULATION §7) for instant, interactive play.
// Same rules as src/pso_rf/optimization/pso.py: continuous state, velocity clamp, absorbing wall, synchronous update,
// strict-improvement pbest/gbest with lowest-index ties. The experiment itself always uses the Python implementation.

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

export class Swarm {
  /**
   * @param {{lower:number[], upper:number[], N:number, w:number, c1:number, c2:number, vmaxFrac:number,
   *          vinitFrac?:number, seed:number, objective:(x:number[])=>number, feedback?:"true"|"constant"|"shuffled"}} o
   */
  constructor(o) {
    Object.assign(this, { vinitFrac: 0.1, feedback: "true", ...o });
    this.D = this.lower.length;
    this.range = this.lower.map((l, d) => this.upper[d] - l);
    this.rng = mulberry32(this.seed);
    this.t = 0; this.history = []; this.evals = 0;
    const r = this.rng;
    this.X = Array.from({ length: this.N }, () => this.lower.map((l, d) => l + r() * this.range[d]));
    this.V = Array.from({ length: this.N }, () => this.range.map((rg) => (r() * 2 - 1) * this.vinitFrac * rg));
    this.F = this.X.map((x) => this.fitness(x));
    this.P = this.X.map((x) => x.slice()); this.Pf = this.F.slice();
    this.g = null; this.gf = -Infinity; this.gIter = 0;
    this.updateGlobal();
    this.record();
  }
  fitness(x) {
    this.evals++;
    if (this.feedback === "constant") return 0;
    if (this.feedback === "shuffled") return this.objective(x.map((v, d) => this.lower[d] + this.upper[d] - v));
    return this.objective(x);
  }
  updateGlobal() {
    let i = 0;
    for (let k = 1; k < this.N; k++) if (this.Pf[k] > this.Pf[i]) i = k;
    if (this.g === null || this.Pf[i] > this.gf) { this.g = this.P[i].slice(); this.gf = this.Pf[i]; this.gIter = this.t; this.improved = true; }
    else this.improved = false;
  }
  step() {
    const { N, D, w, c1, c2, rng } = this, vmax = this.range.map((rg) => this.vmaxFrac * rg);
    this.prevX = this.X.map((x) => x.slice());
    for (let i = 0; i < N; i++) {
      for (let d = 0; d < D; d++) {
        const r1 = rng(), r2 = rng();
        let v = w * this.V[i][d] + c1 * r1 * (this.P[i][d] - this.X[i][d]) + c2 * r2 * (this.g[d] - this.X[i][d]);
        v = Math.max(-vmax[d], Math.min(vmax[d], v));
        let x = this.X[i][d] + v;
        if (x < this.lower[d]) { x = this.lower[d]; v = 0; } else if (x > this.upper[d]) { x = this.upper[d]; v = 0; }
        this.X[i][d] = x; this.V[i][d] = v;
      }
    }
    this.t += 1;
    this.F = this.X.map((x) => this.fitness(x));
    for (let i = 0; i < N; i++) if (this.F[i] > this.Pf[i]) { this.Pf[i] = this.F[i]; this.P[i] = this.X[i].slice(); }
    this.updateGlobal();
    this.record();
  }
  diversity() {
    const Z = this.X.map((x) => x.map((v, d) => (v - this.lower[d]) / this.range[d]));
    const c = Z[0].map((_, d) => Z.reduce((s, z) => s + z[d], 0) / Z.length);
    return Z.reduce((s, z) => s + Math.hypot(...z.map((v, d) => v - c[d])), 0) / Z.length;
  }
  record() {
    const mean = this.F.reduce((a, b) => a + b, 0) / this.N;
    this.history.push({ t: this.t, gf: this.gf, mean, min: Math.min(...this.F), max: Math.max(...this.F), div: this.diversity() });
  }
}
