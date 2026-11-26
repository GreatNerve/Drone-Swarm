import * as THREE from '/vendor/three.module.min.js';

const $ = id => document.getElementById(id);
const canvas = $('view');
const renderer = new THREE.WebGLRenderer({canvas, antialias: true});
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b1116);
scene.fog = new THREE.Fog(0x0b1116, 70, 170);
const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 600);
camera.up.set(0, 0, 1);

const hemi = new THREE.HemisphereLight(0xd8ecff, 0x2b2218, 1.1);
hemi.position.set(0, 0, 1);
scene.add(hemi);
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(30, -25, 45);
scene.add(sun);

// ---------- orbit camera (z up) ----------
const orbit = {azimuth: -2.2, elevation: 0.55, distance: 26, center: new THREE.Vector3(), drag: null};
canvas.addEventListener('contextmenu', e => e.preventDefault());
canvas.addEventListener('pointerdown', e => {
  orbit.drag = {x: e.clientX, y: e.clientY, button: e.button};
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener('pointerup', e => { orbit.drag = null; canvas.releasePointerCapture(e.pointerId); });
canvas.addEventListener('pointermove', e => {
  if (!orbit.drag) return;
  const dx = e.clientX - orbit.drag.x, dy = e.clientY - orbit.drag.y;
  orbit.drag.x = e.clientX; orbit.drag.y = e.clientY;
  if (orbit.drag.button === 2 || e.shiftKey) {
    $('follow').checked = false;
    const scale = orbit.distance * 0.0016;
    const right = new THREE.Vector3(-Math.sin(orbit.azimuth), Math.cos(orbit.azimuth), 0);
    const forward = new THREE.Vector3(-Math.cos(orbit.azimuth), -Math.sin(orbit.azimuth), 0);
    orbit.center.addScaledVector(right, -dx * scale).addScaledVector(forward, dy * scale);
  } else {
    orbit.azimuth -= dx * 0.006;
    orbit.elevation = Math.min(1.45, Math.max(0.08, orbit.elevation + dy * 0.005));
  }
});
canvas.addEventListener('wheel', e => {
  e.preventDefault();
  orbit.distance = Math.min(150, Math.max(4, orbit.distance * Math.exp(e.deltaY * 0.001)));
}, {passive: false});

function placeCamera() {
  const c = Math.cos(orbit.elevation);
  camera.position.set(orbit.center.x + orbit.distance * c * Math.cos(orbit.azimuth),
                      orbit.center.y + orbit.distance * c * Math.sin(orbit.azimuth),
                      orbit.center.z + orbit.distance * Math.sin(orbit.elevation));
  camera.lookAt(orbit.center);
}

function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(canvas);

// ---------- static world ----------
const world = new THREE.Group();
scene.add(world);
let terrainInfo = null;

function heightColor(h, maxH) {
  const t = Math.min(1, Math.max(0, h / Math.max(maxH, 1e-6)));
  const low = [0.16, 0.31, 0.2], mid = [0.45, 0.38, 0.25], high = [0.78, 0.77, 0.72];
  const [a, b, f] = t < 0.55 ? [low, mid, t / 0.55] : [mid, high, (t - 0.55) / 0.45];
  return a.map((v, i) => v + (b[i] - v) * f);
}

function buildWorld(terrain) {
  world.clear();
  terrainInfo = terrain;
  const n = terrain.cells, size = terrain.size, step = size / (n - 1);
  const positions = new Float32Array(n * n * 3), colors = new Float32Array(n * n * 3);
  let maxH = 0;
  for (const row of terrain.heights) for (const h of row) maxH = Math.max(maxH, h);
  for (let iy = 0; iy < n; iy++) for (let ix = 0; ix < n; ix++) {
    const k = (iy * n + ix) * 3, h = terrain.heights[iy][ix];
    positions[k] = -size / 2 + ix * step; positions[k + 1] = -size / 2 + iy * step; positions[k + 2] = h;
    colors.set(heightColor(h, maxH), k);
  }
  const index = [];
  for (let iy = 0; iy < n - 1; iy++) for (let ix = 0; ix < n - 1; ix++) {
    const a = iy * n + ix, b = a + 1, c = a + n, d = c + 1;
    index.push(a, b, d, a, d, c);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geometry.setIndex(index);
  geometry.computeVertexNormals();
  world.add(new THREE.Mesh(geometry, new THREE.MeshLambertMaterial({vertexColors: true})));
  const pillarMaterial = new THREE.MeshLambertMaterial({color: 0x7f8c96});
  for (const p of terrain.obstacles.filter(o => o.kind === 'pillar')) {
    const g = new THREE.CylinderGeometry(p.radius, p.radius * 1.08, p.height, 18);
    g.rotateX(Math.PI / 2);
    const mesh = new THREE.Mesh(g, pillarMaterial);
    mesh.position.set(p.x, p.y, p.base + p.height / 2 - 0.3);
    world.add(mesh);
  }
}

// ---------- dynamic objects ----------
const dynamic = new THREE.Group();
scene.add(dynamic);
const ghostGroup = new THREE.Group(), arrowGroup = new THREE.Group();
dynamic.add(ghostGroup, arrowGroup);
let drones = [], ghosts = [], arrows = [], trails = [];
const TRAIL = 90;

function makeDrone(color) {
  const group = new THREE.Group();
  const dark = new THREE.MeshLambertMaterial({color: 0x2a3138});
  group.add(new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.13, 0.06), dark));
  for (const angle of [Math.PI / 4, -Math.PI / 4]) {
    const arm = new THREE.Mesh(new THREE.BoxGeometry(0.52, 0.03, 0.02), dark);
    arm.rotation.z = angle;
    group.add(arm);
  }
  const rotorMaterial = new THREE.MeshLambertMaterial({color, transparent: true, opacity: 0.75});
  for (const [x, y] of [[1, 1], [1, -1], [-1, 1], [-1, -1]]) {
    const rotor = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.09, 0.012, 18), rotorMaterial);
    rotor.rotation.x = Math.PI / 2;
    rotor.position.set(0.18 * x, 0.18 * y, 0.035);
    group.add(rotor);
  }
  const nose = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.05, 0.04), new THREE.MeshLambertMaterial({color: 0xff5b4a}));
  nose.position.set(0.13, 0, 0);
  group.add(nose);
  const thrust = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3(0, 0, 0.45)]),
                                new THREE.LineBasicMaterial({color}));
  group.add(thrust);
  group.scale.setScalar(1.6);
  return group;
}

function ensureObjects(frame) {
  if (drones.length === frame.drones.length && arrows.length === frame.wind.length) return;
  for (const d of drones) dynamic.remove(d);
  for (const t of trails) dynamic.remove(t.line);
  ghostGroup.clear(); arrowGroup.clear();
  drones = []; ghosts = []; arrows = []; trails = [];
  frame.drones.forEach((_, i) => {
    const color = new THREE.Color().setHSL(i / frame.drones.length, 0.75, 0.6);
    const drone = makeDrone(color);
    dynamic.add(drone); drones.push(drone);
    const ghost = new THREE.Mesh(new THREE.TorusGeometry(0.25, 0.025, 8, 28),
                                 new THREE.MeshBasicMaterial({color: 0xffffff, transparent: true, opacity: 0.35}));
    ghostGroup.add(ghost); ghosts.push(ghost);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(TRAIL * 3), 3));
    geometry.setDrawRange(0, 0);
    const line = new THREE.Line(geometry, new THREE.LineBasicMaterial({color, transparent: true, opacity: 0.5}));
    dynamic.add(line); trails.push({line, count: 0});
  });
  for (let i = 0; i < frame.wind.length; i++) {
    const arrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 1, 0x7fc8ff, 0.45, 0.25);
    arrowGroup.add(arrow); arrows.push(arrow);
  }
}

const targetMarker = new THREE.Mesh(new THREE.OctahedronGeometry(0.22), new THREE.MeshBasicMaterial({color: 0xf2b84b}));
dynamic.add(targetMarker);
const crossing = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 16), new THREE.MeshLambertMaterial({color: 0xd9644f}));
crossing.visible = false;
dynamic.add(crossing);

// ---------- frames ----------
let previous = null, current = null, arrivedAt = 0, interval = 33;
const qa = new THREE.Quaternion(), qb = new THREE.Quaternion();

function applyFrame(alpha) {
  if (!current) return;
  ensureObjects(current);
  const base = previous && previous.drones.length === current.drones.length ? previous : current;
  const centroid = new THREE.Vector3();
  current.drones.forEach((d, i) => {
    const a = base.drones[i], drone = drones[i];
    drone.position.set(a.p[0] + (d.p[0] - a.p[0]) * alpha, a.p[1] + (d.p[1] - a.p[1]) * alpha,
                       a.p[2] + (d.p[2] - a.p[2]) * alpha);
    qa.set(a.q[1], a.q[2], a.q[3], a.q[0]); qb.set(d.q[1], d.q[2], d.q[3], d.q[0]);
    drone.quaternion.slerpQuaternions(qa, qb, alpha);
    centroid.add(drone.position);
    const s = current.slots[i];
    ghosts[i].position.set(s[0], s[1], s[2]);
  });
  centroid.divideScalar(Math.max(1, current.drones.length));
  if ($('follow').checked) orbit.center.lerp(centroid, 0.08);
  targetMarker.position.set(...current.target.p);
  current.wind.forEach((w, i) => {
    const v = new THREE.Vector3(...w.v), speed = v.length();
    arrows[i].position.set(...w.p);
    if (speed > 1e-6) arrows[i].setDirection(v.divideScalar(speed));
    arrows[i].setLength(0.35 + 0.55 * speed, 0.45, 0.25);
  });
  const sphere = current.obstacles.find(o => o.kind === 'sphere');
  crossing.visible = !!sphere;
  if (sphere) { crossing.position.set(...sphere.p); crossing.scale.setScalar(sphere.radius); }
  ghostGroup.visible = $('ghosts').checked;
  arrowGroup.visible = $('arrows').checked;
}

function pushTrails() {
  current.drones.forEach((d, i) => {
    const trail = trails[i], array = trail.line.geometry.attributes.position.array;
    array.copyWithin(3, 0, (TRAIL - 1) * 3);
    array[0] = d.p[0]; array[1] = d.p[1]; array[2] = d.p[2];
    trail.count = Math.min(TRAIL, trail.count + 1);
    trail.line.geometry.setDrawRange(0, trail.count);
    trail.line.geometry.attributes.position.needsUpdate = true;
  });
}

const fmt = (v, d = 2) => v === null || v === undefined ? '-' : Number(v).toFixed(d);

function updateHud(frame) {
  const m = frame.metrics || {}, decision = frame.decision ? frame.decision.decision : null;
  $('clock').textContent = `${fmt(frame.time, 1)} s${frame.paused ? ' (paused)' : ''}`;
  $('region').textContent = m.region || '-';
  $('mode').textContent = decision ? decision.flight_mode : '-';
  $('formation').textContent = decision ? `${decision.formation_preset} ${fmt(decision.formation_scale_m, 1)} m` : '-';
  $('alpha').textContent = decision ? fmt(decision.agility_alpha) : '-';
  $('wind').textContent = `${fmt(m.wind_speed_mps, 1)} m/s`;
  $('tilt').textContent = `${fmt(m.mean_tilt_deg, 1)}° / ${fmt(m.max_tilt_deg, 1)}°`;
  $('speed').textContent = `${fmt(m.speed_mps, 1)} m/s / ${fmt(m.agl_m, 1)} m`;
}

// ---------- reasoning trace ----------
let traceSeen = -1;
function traceItem(entry) {
  const d = entry.decision, li = document.createElement('li');
  if (entry.fault) li.className = 'fault'; else if (entry.clamped.length) li.className = 'clamped';
  const meta = document.createElement('div'); meta.className = 'meta';
  meta.append(Object.assign(document.createElement('span'), {textContent: `#${entry.index} · t=${fmt(entry.time, 1)} s · ${entry.region}`}),
              Object.assign(document.createElement('span'), {textContent: `${d.flight_mode} α=${fmt(d.agility_alpha)}`}));
  const thought = Object.assign(document.createElement('p'), {className: 'thought', textContent: d.thought});
  const fields = Object.assign(document.createElement('div'), {className: 'fields',
    textContent: `${d.formation_preset} ${fmt(d.formation_scale_m, 1)} m · cap ${fmt(d.cruise_speed_cap_mps, 1)} m/s · AGL ${fmt(d.target_agl_m, 1)} m`});
  li.append(meta, thought, fields);
  if (entry.fault || entry.clamped.length) {
    li.append(Object.assign(document.createElement('div'), {className: 'note',
      textContent: entry.fault ? `fault: ${entry.fault}` : `clamped by verifier: ${entry.clamped.join(', ')}`}));
  }
  return li;
}
function addTrace(entries) {
  const list = $('trace');
  for (const entry of entries) {
    if (entry.index <= traceSeen) continue;
    traceSeen = entry.index;
    list.prepend(traceItem(entry));
  }
  while (list.children.length > 8) list.lastChild.remove();
}

async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {method: 'POST', headers: {'Content-Type': 'application/json'},
                                                                body: JSON.stringify(body)});
  if (!response.ok) throw new Error((await response.json()).error || response.statusText);
  return response.json();
}

async function loadWorld() {
  buildWorld(await api('/api/terrain'));
  $('trace').replaceChildren(); traceSeen = -1;
  addTrace((await api('/api/trace')).trace.slice(-8));
  previous = current = null;
  for (const t of trails) { t.count = 0; t.line.geometry.setDrawRange(0, 0); }
}

async function poll() {
  try {
    const frame = await api('/api/frame');
    const now = performance.now();
    if (current) interval = Math.min(100, Math.max(15, 0.8 * interval + 0.2 * (now - arrivedAt)));
    previous = current; current = frame; arrivedAt = now;
    ensureObjects(frame);
    if (!frame.paused) pushTrails();
    updateHud(frame);
    if (frame.decision && frame.decision.index > traceSeen) addTrace([frame.decision]);
  } catch (error) {
    $('badge').textContent = `Viewer server unreachable: ${error.message}`;
  }
  setTimeout(poll, 33);
}

$('reset').onclick = async () => {
  await api('/api/reset', {seed: parseInt($('seed').value, 10) || 0, formation: $('formation-select').value});
  await loadWorld();
};
let paused = false;
$('pause').onclick = async () => {
  paused = !paused;
  await api('/api/pause', {paused});
  $('pause').textContent = paused ? 'Resume' : 'Pause';
};

// ---------- render loop with an FPS counter ----------
let frames = 0, fpsSince = performance.now(), renderMs = 0;
function render(now) {
  applyFrame(Math.min(1, (now - arrivedAt) / interval));
  placeCamera();
  const began = performance.now();
  renderer.render(scene, camera);
  renderMs += performance.now() - began;
  frames++;
  if (now - fpsSince >= 500) {
    $('fps').textContent = Math.round(frames * 1000 / (now - fpsSince));
    window.__swarmv2Fps = frames * 1000 / (now - fpsSince);
    window.__swarmv2Stats = {fps: window.__swarmv2Fps, drawCalls: renderer.info.render.calls,
                             triangles: renderer.info.render.triangles, renderMs: renderMs / frames,
                             textures: renderer.info.memory.textures, geometries: renderer.info.memory.geometries};
    frames = 0; fpsSince = now; renderMs = 0;
  }
  requestAnimationFrame(render);
}

resize();
loadWorld().then(() => { poll(); requestAnimationFrame(render); });
