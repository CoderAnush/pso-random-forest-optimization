// 3-D view of the swarm in the (n_estimators, max_depth, min_samples_split) search space.
import * as THREE from "three";
import { OrbitControls } from "/static/vendor/OrbitControls.js";
import { BOUNDS, HP, fitnessColor, h, fmt } from "./ui.js";

const AXIS_ORDER = [0, 1, 2]; // x = n_estimators, y = max_depth (vertical), z = min_samples_split
const TRAIL = 26;

function norm(v, d) { const [lo, hi] = BOUNDS[d]; return ((v - lo) / (hi - lo)) * 2 - 1; }
function toVec(p) { return new THREE.Vector3(norm(p[0], 0) * 1.25, norm(p[1], 1), norm(p[2], 2)); }

function textSprite(text, { color = "#c3c2b7", size = 26, bold = false } = {}) {
  const c = document.createElement("canvas"), ctx = c.getContext("2d");
  ctx.font = `${bold ? "700" : "500"} ${size}px Inter, Segoe UI, sans-serif`;
  const w = Math.ceil(ctx.measureText(text).width) + 16;
  c.width = w; c.height = size + 14;
  ctx.font = `${bold ? "700" : "500"} ${size}px Inter, Segoe UI, sans-serif`;
  ctx.fillStyle = color; ctx.textBaseline = "middle"; ctx.fillText(text, 8, c.height / 2);
  const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace;
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthWrite: false, transparent: true }));
  const k = 0.0042; sp.scale.set(c.width * k, c.height * k, 1);
  return sp;
}

function discTexture() {
  const c = document.createElement("canvas"); c.width = c.height = 64;
  const g = c.getContext("2d").createRadialGradient(32, 32, 0, 32, 32, 32);
  g.addColorStop(0, "rgba(255,255,255,1)"); g.addColorStop(0.35, "rgba(255,255,255,.85)"); g.addColorStop(1, "rgba(255,255,255,0)");
  const ctx = c.getContext("2d"); ctx.fillStyle = g; ctx.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(c);
}

class Cloud {
  constructor(scene, capacity, size, fixedColor = null) {
    this.cap = capacity; this.n = 0; this.fit = []; this.fixed = fixedColor;
    this.geo = new THREE.BufferGeometry();
    this.pos = new Float32Array(capacity * 3); this.col = new Float32Array(capacity * 3);
    this.geo.setAttribute("position", new THREE.BufferAttribute(this.pos, 3));
    this.geo.setAttribute("color", new THREE.BufferAttribute(this.col, 3));
    this.geo.setDrawRange(0, 0);
    this.mat = new THREE.PointsMaterial({ size, map: discTexture(), vertexColors: true, transparent: true,
      depthWrite: false, blending: THREE.AdditiveBlending, opacity: 0.85, sizeAttenuation: true });
    this.points = new THREE.Points(this.geo, this.mat);
    scene.add(this.points);
  }
  add(cfg, f, range) {
    if (this.n >= this.cap) return;
    const v = toVec(cfg), i = this.n++;
    this.pos.set([v.x, v.y, v.z], i * 3); this.fit[i] = f;
    this.paint(i, range);
    this.geo.attributes.position.needsUpdate = true; this.geo.setDrawRange(0, this.n);
  }
  paint(i, [lo, hi]) {
    let c;
    if (this.fixed) c = this.fixed;
    else { const f = this.fit[i]; c = f == null ? [70, 70, 70] : fitnessColor(hi > lo ? (f - lo) / (hi - lo) : 1); }
    this.col.set([c[0] / 255, c[1] / 255, c[2] / 255], i * 3);
    this.geo.attributes.color.needsUpdate = true;
  }
  repaint(range) { for (let i = 0; i < this.n; i++) this.paint(i, range); }
  clear() { this.n = 0; this.fit = []; this.geo.setDrawRange(0, 0); }
}

export class Swarm3D {
  constructor(container, { autoRotate = true } = {}) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
    container.prepend(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.Fog(0x0b0c0e, 5.5, 11);
    this.camera = new THREE.PerspectiveCamera(42, 1, 0.05, 50);
    this.camera.position.set(3.3, 2.1, 3.6);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    Object.assign(this.controls, { enableDamping: true, dampingFactor: 0.08, autoRotate, autoRotateSpeed: 0.55,
      minDistance: 2.2, maxDistance: 9 });
    this.controls.addEventListener("start", () => { this.controls.autoRotate = false; });
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const key = new THREE.PointLight(0x9fc4ff, 30, 20); key.position.set(3, 4, 3); this.scene.add(key);
    this.buildFrame();
    this.range = [Infinity, -Infinity];
    this.explored = new Cloud(this.scene, 8000, 0.075);
    this.rsCloud = new Cloud(this.scene, 8000, 0.06, [217, 89, 38]);
    this.particles = [];
    this.showTrails = true; this.showVel = true;
    this.buildGbest();
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2(-9, -9);
    this.tip = h("div", { class: "tooltip" }); container.append(this.tip);
    this.renderer.domElement.addEventListener("mousemove", (e) => this.onMove(e));
    this.renderer.domElement.addEventListener("mouseleave", () => { this.mouse.set(-9, -9); this.tip.style.display = "none"; });
    this.ro = new ResizeObserver(() => this.resize()); this.ro.observe(container);
    this.resize();
    this.clock = new THREE.Clock();
    this.alive = true;
    const loop = () => { if (!this.alive) return; this.frame(); this.raf = requestAnimationFrame(loop); };
    loop();
  }

  buildFrame() {
    const box = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(2.5, 2, 2)),
      new THREE.LineBasicMaterial({ color: 0x3a3f47, transparent: true, opacity: 0.9 }));
    this.scene.add(box);
    const grid = new THREE.GridHelper(2.5, 10, 0x2c3138, 0x1d2126); grid.position.y = -1; grid.scale.z = 0.8; this.scene.add(grid);
    const titles = [["n_estimators", new THREE.Vector3(0, -1.28, 1.2)], ["max_depth", new THREE.Vector3(-1.52, 0, 1.12)],
      ["min_samples_split", new THREE.Vector3(1.5, -1.24, 0)]];
    for (const [t, p] of titles) { const s = textSprite(t, { color: "#e6e6e0", size: 30, bold: true }); s.position.copy(p); this.scene.add(s); }
    const ticks = [
      [[50, 2, 10], [-1.25, -1.1, 1.08], "50"], [[200, 2, 10], [1.12, -1.12, 1.12], "200"], [[125, 2, 10], [0, -1.1, 1.08], "125"],
      [[50, 2, 10], [-1.38, -1, 1], "2"], [[50, 20, 10], [-1.38, 1, 1], "20"], [[50, 11, 10], [-1.38, 0, 1], "11"],
      [[200, 2, 2], [1.55, -1.0, -1], "2"], [[200, 2, 10], [1.55, -1.0, 0.85], "10"],
    ];
    for (const [, p, t] of ticks) { const s = textSprite(t, { color: "#898781", size: 24 }); s.position.set(...p); this.scene.add(s); }
  }

  buildGbest() {
    const g = new THREE.Group();
    const core = new THREE.Mesh(new THREE.OctahedronGeometry(0.075), new THREE.MeshStandardMaterial({ color: 0xe66767, emissive: 0xe66767, emissiveIntensity: 1.4 }));
    const ring = new THREE.Mesh(new THREE.TorusGeometry(0.16, 0.008, 8, 48), new THREE.MeshBasicMaterial({ color: 0xe66767, transparent: true, opacity: 0.8 }));
    const ring2 = ring.clone(); ring2.rotation.x = Math.PI / 2;
    const halo = new THREE.Sprite(new THREE.SpriteMaterial({ map: discTexture(), color: 0xe66767, transparent: true, opacity: 0.55, depthWrite: false, blending: THREE.AdditiveBlending }));
    halo.scale.set(0.55, 0.55, 1);
    const drop = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
      new THREE.LineDashedMaterial({ color: 0xe66767, dashSize: 0.04, gapSize: 0.04, transparent: true, opacity: 0.6 }));
    const label = textSprite("gbest", { color: "#e66767", size: 26, bold: true });
    label.position.set(0, 0.2, 0);
    g.add(core, ring, ring2, halo, label);
    g.visible = false;
    this.scene.add(g, drop);
    Object.assign(this, { gbest: g, gbestRing: [ring, ring2], gbestDrop: drop, gbestTarget: new THREE.Vector3() });
  }

  ensureParticles(n) {
    while (this.particles.length < n) {
      const i = this.particles.length;
      const mesh = new THREE.Mesh(new THREE.SphereGeometry(0.045, 20, 14),
        new THREE.MeshStandardMaterial({ color: 0x3987e5, emissive: 0x3987e5, emissiveIntensity: 0.9, roughness: 0.3 }));
      const glow = new THREE.Sprite(new THREE.SpriteMaterial({ map: discTexture(), color: 0x3987e5, transparent: true, opacity: 0.5, depthWrite: false, blending: THREE.AdditiveBlending }));
      glow.scale.set(0.3, 0.3, 1); mesh.add(glow);
      const trailGeo = new THREE.BufferGeometry(); const tp = new Float32Array(TRAIL * 3);
      trailGeo.setAttribute("position", new THREE.BufferAttribute(tp, 3)); trailGeo.setDrawRange(0, 0);
      const trail = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({ color: 0x3987e5, transparent: true, opacity: 0.35 }));
      const velGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
      const vel = new THREE.Line(velGeo, new THREE.LineBasicMaterial({ color: 0xcde2fb, transparent: true, opacity: 0.8 }));
      const head = new THREE.Mesh(new THREE.ConeGeometry(0.018, 0.06, 8), new THREE.MeshBasicMaterial({ color: 0xcde2fb }));
      mesh.visible = trail.visible = vel.visible = head.visible = false;
      mesh.userData = { i };
      this.scene.add(mesh, trail, vel, head);
      this.particles.push({ mesh, glow, trail, trailPts: [], vel, head, cur: new THREE.Vector3(), target: new THREE.Vector3(),
        velTarget: new THREE.Vector3(), info: null, flash: 0, placed: false });
    }
    this.particles.forEach((p, i) => { const on = i < n && p.placed; p.mesh.visible = on; p.trail.visible = on && this.showTrails; });
  }

  /** Move particle i to its new continuous position (smoothly), with the velocity that brought it there. */
  setParticle(i, pos, vel, info) {
    this.ensureParticles(i + 1);
    const p = this.particles[i];
    const target = toVec(pos);
    if (!p.placed) { p.cur.copy(target); p.placed = true; p.mesh.visible = true; }
    p.target.copy(target);
    if (vel) p.velTarget.set(norm(pos[0] + vel[0], 0) * 1.25, norm(pos[1] + vel[1], 1), norm(pos[2] + vel[2], 2)).sub(target);
    p.info = info;
    p.trailPts.push(target.clone()); if (p.trailPts.length > TRAIL) p.trailPts.shift();
    if (info && info.improved) p.flash = 1;
  }
  setGbest(cfg) { this.gbest.visible = true; this.gbestTarget.copy(toVec(cfg)); if (!this.gbestPlaced) { this.gbest.position.copy(this.gbestTarget); this.gbestPlaced = true; } this.gbestPulse = 1; }
  setManual(cfg) {
    if (this.manual) this.scene.remove(this.manual);
    if (!cfg) { this.manual = null; return; }
    const g = new THREE.Group();
    g.add(new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(0.12, 0.12, 0.12)), new THREE.LineDashedMaterial({ color: 0xc3c2b7, dashSize: 0.02, gapSize: 0.015 })).computeLineDistances());
    const l = textSprite("you", { color: "#c3c2b7", size: 24, bold: true }); l.position.set(0, 0.14, 0); g.add(l);
    g.position.copy(toVec(HP.map((k) => cfg[k])));
    this.manual = g; this.scene.add(g);
  }
  addExplored(cfg, f) { this.widen(f); this.explored.add(cfg, f, this.range); }
  addRandom(cfg, f) { this.widen(f); this.rsCloud.add(cfg, f, this.range); }
  widen(f) {
    if (f == null) return;
    const [lo, hi] = this.range;
    if (f < lo || f > hi) { this.range = [Math.min(lo, f), Math.max(hi, f)]; this.needsRepaint = true; }
  }
  setVisible({ trails, velocity, explored, random }) {
    if (trails != null) this.showTrails = trails;
    if (velocity != null) this.showVel = velocity;
    if (explored != null) this.explored.points.visible = explored;
    if (random != null) this.rsCloud.points.visible = random;
  }
  reset() {
    for (const p of this.particles) { p.placed = false; p.trailPts = []; p.mesh.visible = p.trail.visible = p.vel.visible = p.head.visible = false; }
    this.explored.clear(); this.rsCloud.clear(); this.range = [Infinity, -Infinity];
    this.gbest.visible = false; this.gbestPlaced = false; this.gbestDrop.visible = false;
  }

  frame() {
    const dt = Math.min(0.05, this.clock.getDelta()), t = this.clock.elapsedTime, k = 1 - Math.exp(-dt * 7);
    if (this.needsRepaint) { this.explored.repaint(this.range); this.needsRepaint = false; }
    for (const p of this.particles) {
      if (!p.placed) continue;
      p.cur.lerp(p.target, k);
      p.mesh.position.copy(p.cur);
      const fit = p.info?.f, [lo, hi] = this.range;
      const c = fit == null ? [110, 110, 110] : fitnessColor(hi > lo ? (fit - lo) / (hi - lo) : 1);
      const col = new THREE.Color(c[0] / 255, c[1] / 255, c[2] / 255);
      p.mesh.material.color.copy(col); p.mesh.material.emissive.copy(col); p.glow.material.color.copy(col);
      p.flash = Math.max(0, p.flash - dt * 1.6);
      const s = 1 + p.flash * 1.4; p.mesh.scale.setScalar(s); p.glow.material.opacity = 0.45 + p.flash * 0.5;
      const pts = [...p.trailPts.slice(0, -1), p.cur];
      const arr = p.trail.geometry.attributes.position.array;
      pts.forEach((v, j) => arr.set([v.x, v.y, v.z], j * 3));
      p.trail.geometry.setDrawRange(0, pts.length); p.trail.geometry.attributes.position.needsUpdate = true;
      p.trail.visible = this.showTrails && p.mesh.visible;
      const len = p.velTarget.length();
      const showV = this.showVel && len > 0.01 && p.mesh.visible;
      p.vel.visible = p.head.visible = showV;
      if (showV) {
        const end = p.cur.clone().add(p.velTarget);
        p.vel.geometry.attributes.position.setXYZ(0, p.cur.x, p.cur.y, p.cur.z);
        p.vel.geometry.attributes.position.setXYZ(1, end.x, end.y, end.z);
        p.vel.geometry.attributes.position.needsUpdate = true;
        p.head.position.copy(end);
        p.head.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), p.velTarget.clone().normalize());
      }
    }
    if (this.gbest.visible) {
      this.gbest.position.lerp(this.gbestTarget, k);
      this.gbestPulse = Math.max(0, (this.gbestPulse || 0) - dt * 1.2);
      const pulse = 1 + 0.12 * Math.sin(t * 4) + this.gbestPulse * 0.9;
      this.gbest.scale.setScalar(pulse);
      this.gbestRing[0].rotation.z += dt * 1.4; this.gbestRing[1].rotation.y += dt * 1.1;
      const gp = this.gbest.position;
      this.gbestDrop.geometry.setFromPoints([gp.clone(), new THREE.Vector3(gp.x, -1, gp.z)]);
      this.gbestDrop.computeLineDistances(); this.gbestDrop.visible = true;
    }
    this.pick();
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  onMove(e) {
    const r = this.renderer.domElement.getBoundingClientRect();
    this.mouse.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    this.mousePx = [e.clientX - r.left, e.clientY - r.top];
  }
  pick() {
    if (this.mouse.x < -2) return;
    this.raycaster.setFromCamera(this.mouse, this.camera);
    const hits = this.raycaster.intersectObjects(this.particles.filter((p) => p.mesh.visible).map((p) => p.mesh), false);
    if (!hits.length) { this.tip.style.display = "none"; return; }
    const p = this.particles[hits[0].object.userData.i], info = p.info || {};
    this.tip.replaceChildren(
      h("div", {}, h("b", {}, `particle ${hits[0].object.userData.i}`), info.it != null ? ` · iteration ${info.it}` : ""),
      h("div", {}, `config ${fmt.cfg(info.cfg)}`),
      h("div", {}, "fitness ", h("b", {}, fmt.acc(info.f))),
      info.pbest != null ? h("div", { style: { color: "#898781" } }, `personal best ${fmt.acc(info.pbest)}`) : null);
    this.tip.style.display = "block";
    this.tip.style.left = `${this.mousePx[0] + 14}px`; this.tip.style.top = `${this.mousePx[1] + 10}px`;
  }
  resize() {
    const w = this.container.clientWidth || 600, hgt = this.container.clientHeight || 400;
    this.renderer.setSize(w, hgt); this.camera.aspect = w / hgt; this.camera.updateProjectionMatrix();
  }
  dispose() {
    this.alive = false; cancelAnimationFrame(this.raf); this.ro.disconnect(); this.controls.dispose();
    this.renderer.dispose(); this.renderer.domElement.remove(); this.tip.remove();
  }
}
