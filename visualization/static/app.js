import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const API = {
  list: '/api/list_pkls',
  load: '/api/load',
  sample: '/api/sample',
};

const TRAJ = {
  z: 0.5,
};

const state = {
  path: null,
  count: 0,
  idx: 0,
  occMeta: null,
  playing: false,
  playTimer: null,
};

const panels = [];
const frontCanvas = document.getElementById('front-view');
const frontCtx = frontCanvas.getContext('2d');
let frontImage = null;
let frontIntrinsics = null;
let frontExtrinsics = null;

function decodeBytes(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

function decodeF32(b64) {
  const bytes = decodeBytes(b64);
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

function decodeU8(b64) {
  return decodeBytes(b64);
}

function buildPointCloud(payload, classColors) {
  const xyz = decodeF32(payload.xyz_b64);
  const labels = decodeU8(payload.label_b64);
  const n = payload.n;
  const positions = new Float32Array(n * 3);
  const colors = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const x = xyz[i * 3];
    const y = xyz[i * 3 + 1];
    const z = xyz[i * 3 + 2];
    // occ is Z-up (ego: x forward, y left, z up); three.js is Y-up.
    positions[i * 3] = x;
    positions[i * 3 + 1] = z;
    positions[i * 3 + 2] = -y;
    const c = classColors[labels[i]] || [255, 255, 255];
    colors[i * 3] = c[0] / 255;
    colors[i * 3 + 1] = c[1] / 255;
    colors[i * 3 + 2] = c[2] / 255;
  }
  return { positions, colors };
}

function makeSprite() {
  const c = document.createElement('canvas');
  c.width = c.height = 32;
  const g = c.getContext('2d');
  const grad = g.createRadialGradient(16, 16, 0, 16, 16, 16);
  grad.addColorStop(0, 'rgba(255,255,255,1)');
  grad.addColorStop(0.6, 'rgba(255,255,255,0.9)');
  grad.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grad;
  g.fillRect(0, 0, 32, 32);
  return new THREE.CanvasTexture(c);
}

class OccPanel {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setClearColor(0xffffff, 1);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(50, 1, 0.1, 600);
    this.camera.position.set(45, 34, -32);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.target.set(0, 1, 0);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;
    this.sprite = makeSprite();
    this.points = null;
    this.trajGroup = new THREE.Group();
    this.scene.add(this.trajGroup);
    this.scene.add(this._axes());
    this.controls.addEventListener('change', () => {
      syncFrom(this);
    });
  }

  _axes() {
    const g = new THREE.Group();
    const grid = new THREE.GridHelper(80, 20, 0xd5dbe0, 0xe8ecef);
    g.add(grid);
    const len = 6;
    const mat = (c) => new THREE.LineBasicMaterial({ color: c });
    // occ axes drawn in three.js space: X forward (red), Y left (green), Z up (blue)
    const forward = new THREE.BufferGeometry().setFromPoints(
      [new THREE.Vector3(0, 0, 0), new THREE.Vector3(len, 0, 0)]);
    g.add(new THREE.Line(forward, mat(0xd73a49)));
    const left = new THREE.BufferGeometry().setFromPoints(
      [new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, -len)]);
    g.add(new THREE.Line(left, mat(0x2ea043)));
    const up = new THREE.BufferGeometry().setFromPoints(
      [new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, len, 0)]);
    g.add(new THREE.Line(up, mat(0x1f6feb)));
    return g;
  }

  setOcc(payload, classColors) {
    if (this.points) {
      this.scene.remove(this.points);
      this.points.geometry.dispose();
      this.points.material.dispose();
      this.points = null;
    }
    if (!payload) return;
    const pc = buildPointCloud(payload, classColors);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pc.positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(pc.colors, 3));
    const mat = new THREE.PointsMaterial({
      size: 0.5, vertexColors: true, map: this.sprite,
      sizeAttenuation: true, alphaTest: 0.1, transparent: true,
      depthWrite: false,
    });
    this.points = new THREE.Points(geo, mat);
    this.scene.add(this.points);
  }

  setTrajectory(waypoints) {
    while (this.trajGroup.children.length) {
      this.trajGroup.remove(this.trajGroup.children[0]);
    }
    if (!waypoints || !waypoints.length) return;
    const pts = waypoints.map((p) => new THREE.Vector3(p[0], TRAJ.z, -p[1]));
    const line = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(pts),
      new THREE.LineBasicMaterial({ color: 0xff5533, linewidth: 2 }));
    this.trajGroup.add(line);
    const dotGeo = new THREE.SphereGeometry(0.35, 12, 12);
    const dotMat = new THREE.MeshBasicMaterial({ color: 0xff5533 });
    for (const p of pts) {
      const m = new THREE.Mesh(dotGeo, dotMat);
      m.position.copy(p);
      this.trajGroup.add(m);
    }
  }

  resize() {
    const w = this.canvas.clientWidth || 1;
    const h = this.canvas.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  render() {
    this.renderer.render(this.scene, this.camera);
  }
}

function syncFrom(src) {
  for (const p of panels) {
    if (p === src) continue;
    p.camera.position.copy(src.camera.position);
    p.camera.quaternion.copy(src.camera.quaternion);
    p.controls.target.copy(src.controls.target);
  }
}

function animate() {
  requestAnimationFrame(animate);
  for (const p of panels) {
    p.controls.update();
    p.render();
  }
}

function mat4vec(M, p) {
  return [
    M[0][0] * p[0] + M[0][1] * p[1] + M[0][2] * p[2] + M[0][3] * p[3],
    M[1][0] * p[0] + M[1][1] * p[1] + M[1][2] * p[2] + M[1][3] * p[3],
    M[2][0] * p[0] + M[2][1] * p[1] + M[2][2] * p[2] + M[2][3] * p[3],
    M[3][0] * p[0] + M[3][1] * p[1] + M[3][2] * p[2] + M[3][3] * p[3],
  ];
}

function projectPoint(x, y, K, RT) {
  const p = [x, y, TRAJ.z, 1];
  if (K && RT) {
    const cam = mat4vec(RT, p);
    if (cam[2] <= 1e-6) return null;
    const u = K[0][0] * cam[0] + K[0][1] * cam[1] + K[0][2] * cam[2];
    const v = K[1][0] * cam[0] + K[1][1] * cam[1] + K[1][2] * cam[2];
    return [u / cam[2], v / cam[2]];
  }
  if (RT) {
    const uv = mat4vec(RT, p);
    if (uv[2] <= 1e-6) return null;
    return [uv[0] / uv[2], uv[1] / uv[2]];
  }
  return null;
}

function drawFrontView() {
  const w = frontCanvas.clientWidth || 1;
  const h = frontCanvas.clientHeight || 1;
  frontCanvas.width = w;
  frontCanvas.height = h;
  frontCtx.fillStyle = '#ffffff';
  frontCtx.fillRect(0, 0, w, h);

  const ph = document.getElementById('front-placeholder');
  if (!frontImage) {
    ph.classList.remove('hidden');
    return;
  }
  ph.classList.add('hidden');

  const iw = frontImage.naturalWidth || frontImage.width;
  const ih = frontImage.naturalHeight || frontImage.height;
  const scale = Math.min(w / iw, h / ih);
  const dw = iw * scale;
  const dh = ih * scale;
  const ox = (w - dw) / 2;
  const oy = (h - dh) / 2;
  frontCtx.drawImage(frontImage, ox, oy, dw, dh);

  const traj = state.traj;
  if (!traj || !traj.length) return;
  const pts = [];
  for (const p of traj) {
    const uv = projectPoint(p[0], p[1], frontIntrinsics, frontExtrinsics);
    if (!uv) continue;
    pts.push([ox + (uv[0] / iw) * dw, oy + (uv[1] / ih) * dh]);
  }
  if (!pts.length) return;
  frontCtx.strokeStyle = '#ff5533';
  frontCtx.lineWidth = 2;
  frontCtx.beginPath();
  frontCtx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) frontCtx.lineTo(pts[i][0], pts[i][1]);
  frontCtx.stroke();
  frontCtx.fillStyle = '#ff5533';
  for (const [x, y] of pts) {
    frontCtx.beginPath();
    frontCtx.arc(x, y, 4, 0, Math.PI * 2);
    frontCtx.fill();
  }
}

function renderText(text) {
  const el = document.getElementById('text-body');
  el.classList.remove('placeholder');
  if (!text || !text.length) {
    el.innerHTML = '<span class="q-tag">no text</span>';
    return;
  }
  el.innerHTML = text.map((item) => {
    const q = (item.Q || item.question || '') + '';
    const a = (item.A || item.answer || '') + '';
    return '<div class="qa"><div class="q">' + escapeHtml(q) +
      '</div><div class="a">' + escapeHtml(a) + '</div></div>';
  }).join('');
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

function renderLegend(names, colors) {
  const el = document.getElementById('legend-body');
  el.classList.remove('placeholder');
  if (!names || !colors) {
    el.innerHTML = '';
    return;
  }
  el.innerHTML = names.map((name, i) => {
    const c = colors[i] || [255, 255, 255];
    const hex = '#' + c.map((v) => v.toString(16).padStart(2, '0')).join('');
    return '<div class="legend-item"><span class="legend-swatch" style="background:'
      + hex + '"></span>' + escapeHtml(name) + '</div>';
  }).join('');
}

function updateIndicator() {
  const el = document.getElementById('frame-indicator');
  el.textContent = state.count
    ? (state.idx + 1) + ' / ' + state.count : 'no data';
}

function setStatus(msg, isError) {
  const el = document.getElementById('status');
  el.textContent = msg || '';
  el.style.color = isError ? '#cf222e' : 'var(--accent)';
}

async function loadSample(idx) {
  if (!state.path) return;
  state.idx = idx;
  updateIndicator();
  try {
    const url = API.sample + '?path=' + encodeURIComponent(state.path) +
      '&idx=' + idx;
    const res = await fetch(url);
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    state.traj = data.trajectory;
    applySample(data);
    setStatus('token ' + (data.token || '?'));
  } catch (err) {
    setStatus('load error: ' + err.message, true);
  }
}

function applySample(data) {
  const colors = state.occMeta.class_colors;
  panels[0].setOcc(data.occ_pred, colors);
  panels[0].setTrajectory(data.trajectory);

  const fu = data.occ_forecasting;
  if (fu && fu.length >= 4) {
    panels[1].setOcc(fu[1], colors);
    panels[2].setOcc(fu[2], colors);
    panels[3].setOcc(fu[3], colors);
  } else {
    panels[1].setOcc(null, colors);
    panels[2].setOcc(null, colors);
    panels[3].setOcc(null, colors);
  }
  for (const p of panels.slice(1)) p.setTrajectory(data.trajectory);

  frontIntrinsics = data.intrinsics;
  frontExtrinsics = data.extrinsics;
  const img = new Image();
  img.onload = () => { frontImage = img; drawFrontView(); };
  img.onerror = () => { frontImage = null; drawFrontView(); };
  frontImage = null;
  if (data.front_image) img.src = data.front_image;
  else drawFrontView();

  renderText(data.text);
}

async function loadPkl(path) {
  if (!path) return;
  setStatus('loading ' + path + ' ...');
  try {
    const res = await fetch(API.load + '?path=' + encodeURIComponent(path));
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    state.path = data.path;
    state.count = data.count;
    state.occMeta = data.occ_meta;
    renderLegend(data.occ_meta.class_names, data.occ_meta.class_colors);
    state.idx = 0;
    document.getElementById('pkl-path').value = data.path;
    updateIndicator();
    await loadSample(0);
  } catch (err) {
    setStatus('load error: ' + err.message, true);
  }
}

function stopPlaying() {
  state.playing = false;
  if (state.playTimer) { clearInterval(state.playTimer); state.playTimer = null; }
  document.getElementById('btn-play').innerHTML = '&#9654; Play';
}

function togglePlay() {
  if (state.playing) { stopPlaying(); return; }
  if (!state.count) return;
  state.playing = true;
  document.getElementById('btn-play').innerHTML = '&#10074;&#10074; Pause';
  state.playTimer = setInterval(() => {
    loadSample((state.idx + 1) % state.count);
  }, 1000);
}

async function populateList() {
  try {
    const res = await fetch(API.list);
    const data = await res.json();
    const sel = document.getElementById('pkl-list');
    sel.innerHTML = '<option value="">-- choose a pkl --</option>';
    for (const f of (data.files || [])) {
      const o = document.createElement('option');
      o.value = f.path;
      o.textContent = f.path;
      sel.appendChild(o);
    }
  } catch (e) {
    console.warn('pkl listing failed', e);
  }
}

function bindUI() {
  document.getElementById('btn-load').addEventListener('click', () => {
    const path = document.getElementById('pkl-path').value.trim();
    loadPkl(path);
  });
  document.getElementById('pkl-list').addEventListener('change', (e) => {
    if (e.target.value) {
      document.getElementById('pkl-path').value = e.target.value;
      loadPkl(e.target.value);
    }
  });
  document.getElementById('pkl-path').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      loadPkl(document.getElementById('pkl-path').value.trim());
    }
  });
  document.getElementById('btn-prev').addEventListener('click', () => {
    stopPlaying();
    if (state.count) loadSample((state.idx - 1 + state.count) % state.count);
  });
  document.getElementById('btn-next').addEventListener('click', () => {
    stopPlaying();
    if (state.count) loadSample((state.idx + 1) % state.count);
  });
  document.getElementById('btn-play').addEventListener('click', togglePlay);
}

function bindResize() {
  const canvases = [
    'occ-current', 'occ-fu1', 'occ-fu2', 'occ-fu3',
  ];
  for (const id of canvases) {
    const el = document.getElementById(id);
    const ro = new ResizeObserver(() => {
      panels.forEach((p) => p.resize());
      drawFrontView();
    });
    ro.observe(el);
  }
  const frontRo = new ResizeObserver(() => drawFrontView());
  frontRo.observe(frontCanvas);
}

function init() {
  const ids = ['occ-current', 'occ-fu1', 'occ-fu2', 'occ-fu3'];
  for (const id of ids) {
    panels.push(new OccPanel(document.getElementById(id)));
  }
  bindUI();
  bindResize();
  populateList();
  animate();
  const prefill = document.getElementById('pkl-path').value;
  if (!prefill) {
    document.getElementById('pkl-path').value =
      '/home/adt/sparseocc_occ_baseline.pkl';
  }
}

init();

window.__viz = { panels, state };
