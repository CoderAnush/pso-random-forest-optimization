// 3-D terrain of a fitness landscape over (n_estimators, max_depth) with the swarm flying over it.
import * as THREE from "three";
import { OrbitControls } from "/static/vendor/OrbitControls.js";
import { LOWER, UPPER } from "./landscapes.js";
import { fitnessColor } from "./ui.js";

const SX = 1.5, SZ = 1.05, HGT = 0.95;
const wx = (n) => ((n - LOWER[0]) / (UPPER[0] - LOWER[0]) * 2 - 1) * SX;
const wz = (d) => -((d - LOWER[1]) / (UPPER[1] - LOWER[1]) * 2 - 1) * SZ;

function label(text, color = "#c3c2b7", size = 26) {
  const c = document.createElement("canvas"), ctx = c.getContext("2d");
  ctx.font = `600 ${size}px Inter, Segoe UI, sans-serif`;
  c.width = Math.ceil(ctx.measureText(text).width) + 12; c.height = size + 12;
  ctx.font = `600 ${size}px Inter, Segoe UI, sans-serif`; ctx.fillStyle = color; ctx.textBaseline = "middle"; ctx.fillText(text, 6, c.height / 2);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: t, depthWrite: false, transparent: true }));
  s.scale.set(c.width * 0.004, c.height * 0.004, 1); return s;
}

export class Terrain3D {
  constructor(container) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
    container.append(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.05, 40);
    this.camera.position.set(0.9, 2.9, 3.9);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    Object.assign(this.controls, { enableDamping: true, autoRotate: true, autoRotateSpeed: 0.4, minDistance: 1.8, maxDistance: 7 });
    this.controls.target.set(0, 0.25, 0);
    this.controls.addEventListener("start", () => { this.controls.autoRotate = false; });
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.32));
    const sun = new THREE.DirectionalLight(0xffffff, 0.95); sun.position.set(2, 4, 1.5); this.scene.add(sun);
    const rim = new THREE.DirectionalLight(0x86b6ef, 0.6); rim.position.set(-2, 1, -2); this.scene.add(rim);
    for (const [t, p] of [["n_estimators →", [0, -0.05, SZ + 0.25]], ["max_depth →", [-SX - 0.35, -0.05, 0]]]) {
      const s = label(t, "#e6e6e0", 28); s.position.set(...p); this.scene.add(s);
    }
    this.particles = []; this.ghosts = [];
    const gb = new THREE.Group();
    gb.add(new THREE.Mesh(new THREE.OctahedronGeometry(0.05), new THREE.MeshStandardMaterial({ color: 0xe66767, emissive: 0xe66767, emissiveIntensity: 1.3 })));
    this.beam = new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.006, 1, 8), new THREE.MeshBasicMaterial({ color: 0xe66767, transparent: true, opacity: 0.5 }));
    this.scene.add(gb, this.beam); this.gbest = gb;
    this.star = new THREE.Mesh(new THREE.TorusGeometry(0.06, 0.012, 8, 32), new THREE.MeshBasicMaterial({ color: 0xffffff }));
    this.star.rotation.x = Math.PI / 2; this.scene.add(this.star);
    this.ro = new ResizeObserver(() => this.resize()); this.ro.observe(container); this.resize();
    this.alive = true; this.clock = new THREE.Clock();
    const loop = () => { if (!this.alive) return; this.frame(); this.raf = requestAnimationFrame(loop); }; loop();
  }
  height(f) { return ((f - this.lo) / Math.max(1e-9, this.hi - this.lo)) * HGT; }
  setLandscape(L) {
    this.L = L; this.lo = L.lo; this.hi = L.best;
    if (this.mesh) { this.scene.remove(this.mesh, this.wire); this.mesh.geometry.dispose(); }
    let nx, nz, value;
    if (L.discrete) { nx = L.grid.ns.length; nz = L.grid.ds.length; value = (i, j) => [L.grid.ns[i], L.grid.ds[j], L.grid.table[j][i]]; }
    else { nx = nz = 90; value = (i, j) => { const n = LOWER[0] + (i / (nx - 1)) * (UPPER[0] - LOWER[0]), d = LOWER[1] + (j / (nz - 1)) * (UPPER[1] - LOWER[1]); return [n, d, L.f([n, d])]; }; }
    const geo = new THREE.BufferGeometry(), pos = [], col = [], idx = [];
    for (let j = 0; j < nz; j++) for (let i = 0; i < nx; i++) {
      const [n, d, f] = value(i, j); pos.push(wx(n), this.height(f), wz(d));
      const c = fitnessColor((f - this.lo) / Math.max(1e-9, this.hi - this.lo)); col.push(c[0] / 255, c[1] / 255, c[2] / 255);
    }
    for (let j = 0; j < nz - 1; j++) for (let i = 0; i < nx - 1; i++) { const a = j * nx + i, b = a + 1, c = a + nx, e = c + 1; idx.push(a, c, b, b, c, e); }
    geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3)); geo.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
    geo.setIndex(idx); geo.computeVertexNormals();
    this.mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.85, metalness: 0.0, side: THREE.DoubleSide, flatShading: L.discrete }));
    this.wire = new THREE.LineSegments(new THREE.WireframeGeometry(geo), new THREE.LineBasicMaterial({ color: 0x0b0c0e, transparent: true, opacity: L.discrete ? 0.35 : 0.08 }));
    this.scene.add(this.mesh, this.wire);
    this.star.position.set(wx(L.arg[0]), this.height(L.best) + 0.02, wz(L.arg[1]));
  }
  surf(x) { return this.height(this.L.f(x)); }
  setSwarm(disp, P, g) {
    while (this.particles.length < disp.length) {
      const m = new THREE.Mesh(new THREE.SphereGeometry(0.03, 16, 12), new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0x3987e5, emissiveIntensity: 0.9 }));
      const gh = new THREE.Mesh(new THREE.SphereGeometry(0.014, 8, 6), new THREE.MeshBasicMaterial({ color: 0x86b6ef, transparent: true, opacity: 0.6 }));
      this.scene.add(m, gh); this.particles.push(m); this.ghosts.push(gh);
    }
    this.particles.forEach((m, i) => { m.visible = i < disp.length; this.ghosts[i].visible = i < disp.length; });
    disp.forEach((x, i) => {
      this.particles[i].position.set(wx(x[0]), this.surf(x) + 0.08, wz(x[1]));
      this.ghosts[i].position.set(wx(P[i][0]), this.surf(P[i]) + 0.03, wz(P[i][1]));
    });
    if (g) {
      const y = this.surf(g); this.gbest.position.set(wx(g[0]), y + 0.16, wz(g[1]));
      this.beam.position.set(wx(g[0]), y + 0.6, wz(g[1])); this.beam.scale.y = 1.0;
    }
  }
  frame() {
    const t = this.clock.getElapsedTime();
    this.gbest.rotation.y = t * 1.5; this.gbest.scale.setScalar(1 + 0.15 * Math.sin(t * 4));
    this.star.rotation.z = t;
    this.controls.update(); this.renderer.render(this.scene, this.camera);
  }
  resize() { const w = this.container.clientWidth || 600, h = this.container.clientHeight || 400; this.renderer.setSize(w, h); this.camera.aspect = w / h; this.camera.updateProjectionMatrix(); }
  dispose() { this.alive = false; cancelAnimationFrame(this.raf); this.ro.disconnect(); this.controls.dispose(); this.renderer.dispose(); this.renderer.domElement.remove(); }
}
