/**
 * 3D-превʼю DDA: частинка, диполі, моменти p, поле, хвиля k, осі XYZ.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const MATERIAL_COLORS = { Au: '#e8c547', Ag: '#e8eef5', Al: '#b8c8d8' };
const MATERIAL_METAL = { Au: 0.92, Ag: 0.88, Al: 0.72 };
const ENV_COLORS = { Air: 0x5c8fd6, Water: 0x1e88e5, Glass: 0x66bb6a, Custom: 0x9e9e9e };

let scene, camera, renderer, controls;
let previewGroup, axesGroup, lightMarker;
let layerParticle, layerDipoles, layerMoments, layerWave, layerField, layerEnv;
let previewTimer = null;
let previewRequestId = 0;
let lastPreviewData = null;
let animEnabled = false;
let animPhase = 0;
let momentArrows = [];

export function initDdaPreview3d() {
    const canvas = document.getElementById('preview-canvas');
    if (!canvas) return;

    const w = canvas.clientWidth || 640;
    const h = 420;
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0e14);
    camera = new THREE.PerspectiveCamera(45, w / h, 0.5, 12000);
    camera.position.set(140, 100, 160);

    renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(w, h, false);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    scene.add(new THREE.AmbientLight(0x8899aa, 0.55));
    const sun = new THREE.DirectionalLight(0xffffff, 1.15);
    sun.position.set(80, 120, 60);
    scene.add(sun);

    previewGroup = new THREE.Group();
    layerParticle = new THREE.Group();
    layerDipoles = new THREE.Group();
    layerMoments = new THREE.Group();
    layerWave = new THREE.Group();
    layerField = new THREE.Group();
    layerEnv = new THREE.Group();
    lightMarker = new THREE.Group();
    previewGroup.add(layerEnv, layerField, layerParticle, layerDipoles, layerMoments, layerWave, lightMarker);
    scene.add(previewGroup);

    axesGroup = new THREE.Group();
    const axLen = 35;
    axesGroup.add(new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 0, 0), axLen, 0xff5555, 6, 3));
    axesGroup.add(new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 0), axLen, 0x55ff55, 6, 3));
    axesGroup.add(new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 0, 0), axLen, 0x5555ff, 6, 3));
    scene.add(axesGroup);

    function animate() {
        requestAnimationFrame(animate);
        if (animEnabled && momentArrows.length) {
            animPhase += 0.08;
            const s = 0.55 + 0.45 * Math.sin(animPhase);
            momentArrows.forEach((arr) => {
                const base = arr.userData.baseLen || 1;
                arr.setLength(base * s);
            });
        }
        controls.update();
        renderer.render(scene, camera);
    }
    animate();

    window.addEventListener('resize', onResize);
    document.getElementById('btn-refresh-preview')?.addEventListener('click', updatePreview);
    document.getElementById('animate-dipoles')?.addEventListener('change', (e) => {
        animEnabled = e.target.checked;
    });
    document.querySelectorAll('input[name="view_mode"]').forEach((el) => {
        el.addEventListener('change', () => applyViewMode());
    });
    ['show_dipoles', 'show_electric_field', 'show_near_field'].forEach((id) => {
        document.getElementById(id)?.addEventListener('change', () => {
            if (lastPreviewData) {
                renderScene(lastPreviewData);
            } else {
                schedulePreviewUpdate();
            }
        });
    });

    const form = document.getElementById('sim-form');
    form?.addEventListener('input', (e) => {
        if (e.target.closest('.preview-geom, .preview-visual')) schedulePreviewUpdate();
    });
    form?.addEventListener('change', (e) => {
        if (e.target.id === 'wavelength_min_nm') syncPreviewWavelengthFromMin();
        if (e.target.closest('.preview-geom, .preview-visual, #environment, #material')) schedulePreviewUpdate();
    });
    document.getElementById('material')?.addEventListener('change', () => schedulePreviewUpdate());
    document.getElementById('preview_wavelength_nm')?.addEventListener('input', () => schedulePreviewUpdate());
    document.getElementById('preview_wavelength_nm')?.addEventListener('change', () => schedulePreviewUpdate());
    document.getElementById('environment')?.addEventListener('change', toggleAmbientIndex);
    toggleAmbientIndex();
    updatePreview();
}

function onResize() {
    const canvas = document.getElementById('preview-canvas');
    if (!renderer || !canvas) return;
    const w = canvas.clientWidth;
    const h = 420;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h, false);
}

function syncPreviewWavelengthFromMin() {
    const minEl = document.getElementById('wavelength_min_nm');
    const previewEl = document.getElementById('preview_wavelength_nm');
    if (minEl && previewEl) previewEl.value = minEl.value;
}

function getPreviewWavelengthNm() {
    const el = document.getElementById('preview_wavelength_nm');
    const v = parseFloat(el?.value);
    if (!Number.isNaN(v) && v > 0) return v;
    const minV = parseFloat(document.getElementById('wavelength_min_nm')?.value);
    return !Number.isNaN(minV) && minV > 0 ? minV : 500;
}

function formParams() {
    const form = document.getElementById('sim-form');
    const data = new FormData(form);
    const params = new URLSearchParams();
    for (const [k, v] of data.entries()) params.set(k, v);
    params.set('preview_wavelength_nm', String(getPreviewWavelengthNm()));
    if (!document.getElementById('show_near_field')?.checked) params.set('show_near_field', 'off');
    return params;
}

function readVisualParams() {
    return {
        environment: document.getElementById('environment')?.value || 'Air',
        ambient_index: parseFloat(document.getElementById('ambient_index')?.value) || 1,
        polarization: document.getElementById('polarization')?.value || 'X',
        show_dipoles: document.getElementById('show_dipoles')?.checked ?? true,
        show_electric_field: document.getElementById('show_electric_field')?.checked ?? true,
        show_near_field: document.getElementById('show_near_field')?.checked ?? false,
    };
}

function getViewMode() {
    return document.querySelector('input[name="view_mode"]:checked')?.value || 'full';
}

function resetParticleOpacity() {
    layerParticle?.children.forEach((c) => {
        if (c.material && 'opacity' in c.material) c.material.opacity = 0.35;
    });
}

function applyViewMode() {
    const mode = getViewMode();
    const v = readVisualParams();
    const show = (layer, on) => { if (layer) layer.visible = on; };

    resetParticleOpacity();
    show(layerParticle, false);
    show(layerDipoles, false);
    show(layerMoments, false);
    show(layerWave, false);
    show(layerField, false);
    show(layerEnv, false);
    show(lightMarker, false);

    const showField = v.show_near_field && (lastPreviewData?.near_field?.length > 0);

    switch (mode) {
        case 'particle':
            show(layerParticle, true);
            show(layerEnv, true);
            break;
        case 'dipoles':
            show(layerParticle, true);
            layerParticle.children.forEach((c) => { if (c.material) c.material.opacity = 0.08; });
            show(layerDipoles, v.show_dipoles);
            break;
        case 'moments':
            show(layerParticle, true);
            show(layerMoments, true);
            break;
        case 'field':
            show(layerParticle, true);
            layerParticle.children.forEach((c) => { if (c.material) c.material.opacity = 0.06; });
            show(layerField, showField);
            show(layerWave, v.show_electric_field);
            show(lightMarker, true);
            break;
        case 'full':
        default:
            show(layerParticle, true);
            show(layerDipoles, v.show_dipoles);
            show(layerMoments, true);
            show(layerWave, v.show_electric_field);
            show(layerField, showField);
            show(layerEnv, true);
            show(lightMarker, true);
            break;
    }
}

function vecFromArray(a) {
    return new THREE.Vector3(a[0] || 0, a[1] || 0, a[2] || 1);
}

function applyParticleRotation(group, axis, thetaDeg, phiDeg) {
    group.rotation.set(0, 0, 0);
    const t = THREE.MathUtils.degToRad(parseFloat(thetaDeg) || 0);
    const p = THREE.MathUtils.degToRad(parseFloat(phiDeg) || 0);
    if (axis === 'Y') group.rotateY(t);
    else if (axis === 'Z') group.rotateZ(t);
    else group.rotateX(t);
    group.rotateZ(p);
}

function disposeGroup(group) {
    while (group.children.length) {
        const obj = group.children[0];
        group.remove(obj);
        obj.traverse?.((c) => {
            if (c.geometry) c.geometry.dispose();
            if (c.material) {
                const m = c.material;
                if (Array.isArray(m)) m.forEach((x) => x.dispose());
                else m.dispose();
            }
        });
    }
}

function clearLayers() {
    [layerParticle, layerDipoles, layerMoments, layerWave, layerField, layerEnv, lightMarker].forEach(disposeGroup);
    momentArrows = [];
}

function createShapeGeometry(data) {
    const r = parseFloat(data.radius_nm) || 50;
    const ar = Math.max(parseFloat(data.aspect_ratio) || 1, 0.1);
    let geom;
    if (data.shape === 'sphere') geom = new THREE.SphereGeometry(r, 40, 32);
    else if (data.shape === 'ellipsoid') {
        geom = new THREE.SphereGeometry(r, 40, 32);
        geom.scale(1, ar, ar);
    } else if (data.shape === 'cube') geom = new THREE.BoxGeometry(2 * r, 2 * r, 2 * r);
    else {
        geom = new THREE.CylinderGeometry(r, r, 2 * r * ar, 32);
        geom.rotateX(Math.PI / 2);
    }
    return { geom, r, ar };
}

function heatColor(t) {
    const x = Math.max(0, Math.min(1, t));
    const g = Math.pow(x, 0.55);
    const c = new THREE.Color();
    if (g < 0.25) {
        c.setRGB(0.05, 0.12 + g * 1.6, 0.55 + g * 0.8);
    } else if (g < 0.55) {
        const u = (g - 0.25) / 0.3;
        c.setRGB(u * 0.2, 0.55 + u * 0.45, 1.0 - u * 0.85);
    } else if (g < 0.8) {
        const u = (g - 0.55) / 0.25;
        c.setRGB(0.2 + u * 0.85, 1.0 - u * 0.25, 0.15 * (1 - u));
    } else {
        const u = (g - 0.8) / 0.2;
        c.setRGB(1.0, 0.35 * (1 - u), u * 0.15);
    }
    return c;
}

function addParticle(data) {
    const { geom, r, ar } = createShapeGeometry(data);
    const matName = data.material || 'Au';
    const color = new THREE.Color(data.color || MATERIAL_COLORS[matName] || '#e8c547');
    const epsIm = Math.abs(parseFloat(data.epsilon_imag) || 0);
    const resp = Math.min(2.5, parseFloat(data.response_strength) || 0.5);
    const mat = new THREE.MeshPhysicalMaterial({
        color,
        metalness: MATERIAL_METAL[matName] ?? 0.75,
        roughness: 0.2 + Math.min(0.4, epsIm * 0.015),
        transparent: true,
        opacity: 0.28 + Math.min(0.2, resp * 0.08),
        clearcoat: 0.35,
        emissive: color.clone().multiplyScalar(0.05 + resp * 0.25),
    });
    const mesh = new THREE.Mesh(geom, mat);
    layerParticle.add(mesh);
    layerParticle.add(new THREE.LineSegments(
        new THREE.EdgesGeometry(geom),
        new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.85 })
    ));
    return r * Math.max(ar, 1) * 2.3;
}

function dipoleVisualSize(data) {
    const spacing = parseFloat(data.spacing_nm) || parseFloat(data.limits?.spacing_nm) || 5;
    const r = parseFloat(data.radius_nm) || 50;
    return Math.max(0.35, Math.min(spacing * 0.22, r * 0.045));
}

function addDipoleLattice(data) {
    if (!data.positions?.length) return;
    const d = dipoleVisualSize(data);
    const n = data.positions.length;

    if (n <= 600) {
        const box = new THREE.BoxGeometry(d, d, d);
        const mat = new THREE.MeshBasicMaterial({ color: 0xffc107, transparent: true, opacity: 0.55 });
        const mesh = new THREE.InstancedMesh(box, mat, n);
        const m = new THREE.Matrix4();
        const p = new THREE.Vector3();
        for (let i = 0; i < n; i++) {
            p.set(data.positions[i][0], data.positions[i][1], data.positions[i][2]);
            m.setPosition(p);
            mesh.setMatrixAt(i, m);
        }
        mesh.instanceMatrix.needsUpdate = true;
        layerDipoles.add(mesh);
    } else {
        const flat = [];
        for (const pt of data.positions) flat.push(pt[0], pt[1], pt[2]);
        const geom = new THREE.BufferGeometry();
        geom.setAttribute('position', new THREE.Float32BufferAttribute(flat, 3));
        layerDipoles.add(new THREE.Points(geom, new THREE.PointsMaterial({
            color: 0xffc107,
            size: Math.max(1.2, d * 0.9),
            sizeAttenuation: true,
        })));
    }
}

function addMomentArrows(data) {
    const list = data.dipole_moments || [];
    const r = parseFloat(data.radius_nm) || 50;
    const pMax = Math.max(parseFloat(data.max_dipole_moment) || 0, 1e-30);

    list.forEach((item) => {
        const pos = new THREE.Vector3(item.pos[0], item.pos[1], item.pos[2]);
        const dir = new THREE.Vector3(item.dir[0], item.dir[1], item.dir[2]).normalize();
        const magShape = item.mag_rel ?? 1;
        const magAbs = item.mag_abs ?? 0;
        const len = r * 0.42 * (magAbs / pMax) * (0.25 + 0.75 * magShape);
        const origin = pos.clone().sub(dir.clone().multiplyScalar(len * 0.5));
        const t = Math.min(1, magAbs / pMax);
        const col = new THREE.Color().setHSL(0.08 - t * 0.08, 0.95, 0.35 + t * 0.35);
        const arrow = new THREE.ArrowHelper(dir, origin, len, col.getHex(), r * 0.045, r * 0.03);
        arrow.userData.baseLen = len;
        layerMoments.add(arrow);
        momentArrows.push(arrow);
    });
}

function polDirection(pol) {
    if (pol === 'Y') return new THREE.Vector3(0, 1, 0);
    if (pol === 'Z') return new THREE.Vector3(0, 0, 1);
    return new THREE.Vector3(1, 0, 0);
}

function addWaveAndPol(data, visual) {
    const r = parseFloat(data.radius_nm) || 50;
    const wl = parseFloat(data.preview_wavelength_nm) || getPreviewWavelengthNm();
    const pol = polDirection(visual.polarization || data.polarization);
    const polLen = r * 2.5;
    layerWave.add(new THREE.ArrowHelper(
        pol, new THREE.Vector3(0, 0, 0), polLen, 0xffeb3b, r * 0.12, r * 0.08
    ));

    const kDir = vecFromArray(data.wave_vector).normalize();
    const dist = r * 4.8;
    const src = kDir.clone().multiplyScalar(-dist);
    const pathLen = Math.max(dist * 2.2, wl * 2.5);
    const nWaves = Math.min(14, Math.max(4, Math.round(pathLen / wl)));
    for (let i = 0; i < nWaves; i++) {
        const s = (i / Math.max(nWaves - 1, 1)) * pathLen;
        const origin = src.clone().add(kDir.clone().multiplyScalar(s));
        layerWave.add(new THREE.ArrowHelper(
            kDir, origin, Math.min(r * 0.45, wl * 0.35), 0x69f0ae, r * 0.07, r * 0.05
        ));
    }

    const amp = r * 0.18;
    const steps = 128;
    const pts = [];

    for (let i = 0; i <= steps; i++) {
        const s = (i / steps) * pathLen;
        const base = src.clone().add(kDir.clone().multiplyScalar(s));
        const phase = (2 * Math.PI * s) / Math.max(wl, 50);
        const offset = pol.clone().multiplyScalar(amp * Math.sin(phase));
        pts.push(base.add(offset));
    }

    layerWave.add(new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(pts),
        new THREE.LineBasicMaterial({ color: 0x69f0ae, transparent: true, opacity: 0.8 })
    ));
}

function addLightSourceMarker(data) {
    const r = parseFloat(data.radius_nm) || 50;
    const kDir = vecFromArray(data.wave_vector).normalize();
    const dist = r * 4.8;
    const pos = kDir.clone().multiplyScalar(-dist);
    const sphere = new THREE.Mesh(
        new THREE.SphereGeometry(r * 0.18, 16, 12),
        new THREE.MeshBasicMaterial({ color: 0xfff59d })
    );
    sphere.position.copy(pos);
    lightMarker.add(sphere);
    lightMarker.add(new THREE.ArrowHelper(
        kDir, pos, dist * 0.35, 0xfff176, r * 0.1, r * 0.07
    ));
}

function addNearField(data) {
    const pts = data.near_field || [];
    if (!pts.length) return;
    const r = parseFloat(data.radius_nm) || 50;
    const pointSize = Math.max(1.6, Math.min(4.0, r * 0.034));

    const sorted = pts.map((p) => p.intensity).sort((a, b) => a - b);
    const pLo = sorted[Math.floor(sorted.length * 0.06)] ?? 0;
    const pHi = sorted[Math.floor(sorted.length * 0.94)] ?? 1;
    const span = Math.max(pHi - pLo, 1e-6);

    const positions = [];
    const colors = [];
    pts.forEach((pt) => {
        positions.push(pt.x, pt.y, pt.z);
        let t = (pt.intensity - pLo) / span;
        t = Math.max(0, Math.min(1, t));
        t = Math.pow(t, 0.5);
        const c = heatColor(t);
        colors.push(c.r, c.g, c.b);
    });
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geom.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    const cloud = new THREE.Points(geom, new THREE.PointsMaterial({
        size: pointSize,
        sizeAttenuation: true,
        vertexColors: true,
        transparent: true,
        opacity: 0.88,
        depthWrite: false,
        depthTest: true,
    }));
    cloud.renderOrder = 5;
    layerField.add(cloud);
}

function addEnvironment(data, visual) {
    const r = parseFloat(data.radius_nm) || 50;
    const ar = Math.max(parseFloat(data.aspect_ratio) || 1, 0.1);
    const shellR = r * Math.max(ar, 1) * 3.2;
    const col = ENV_COLORS[visual.environment] ?? ENV_COLORS.Air;
    layerEnv.add(new THREE.Mesh(
        new THREE.SphereGeometry(shellR, 36, 28),
        new THREE.MeshPhongMaterial({
            color: col, transparent: true, opacity: 0.05, side: THREE.BackSide, depthWrite: false,
        })
    ));
}

function fitCamera(sizeNm) {
    const dist = Math.max(sizeNm * 3.0, 90);
    camera.position.set(dist * 0.85, dist * 0.65, dist);
    controls.target.set(0, 0, 0);
    controls.update();
}

function renderScene(data) {
    const visual = { ...readVisualParams(), ...data };
    clearLayers();
    const sizeNm = addParticle(data);
    addDipoleLattice(data);
    addMomentArrows(data);
    addWaveAndPol(data, visual);
    addLightSourceMarker(data);
    if (visual.show_near_field && data.near_field?.length) addNearField(data);
    addEnvironment(data, visual);
    applyParticleRotation(previewGroup, data.orientation_axis || 'X', data.theta_deg, data.phi_deg);
    fitCamera(sizeNm);
    applyViewMode();
}

export function applyLimitsFromPreview(data) {
    const lim = data.limits || {};
    const eff = data.effective || {};
    const setNum = (id, val) => {
        const el = document.getElementById(id);
        if (el && val !== undefined && !Number.isNaN(val)) el.value = val;
    };
    setNum('radius_nm', eff.radius_nm);
    setNum('aspect_ratio', eff.aspect_ratio);
    setNum('dipole_spacing_nm', eff.dipole_spacing_nm);
    setNum('num_dipoles', eff.num_dipoles);
    const nd = document.getElementById('num_dipoles');
    const ds = document.getElementById('dipole_spacing_nm');
    const rn = document.getElementById('radius_nm');
    if (nd && lim.num_dipoles_max) { nd.max = lim.num_dipoles_max; nd.min = lim.num_dipoles_min || 1; }
    if (ds && lim.spacing_max_nm) { ds.min = lim.spacing_min_nm || 0.5; ds.max = lim.spacing_max_nm; }
    if (rn) { rn.min = lim.radius_min_nm || 2; rn.max = lim.radius_max_nm || 500; }
    const hintD = document.getElementById('hint-dipoles');
    const hintS = document.getElementById('hint-spacing');
    if (hintD && lim.lattice_count !== undefined) {
        hintD.innerHTML = `λ<sub>3D</sub>=${data.preview_wavelength_nm?.toFixed(0) ?? '—'} нм · |α|=${data.polarizability_abs_um3?.toExponential(2) ?? '—'} μm³ · відгук=${data.response_strength?.toFixed(2) ?? '—'}`;
    }
    if (hintS && lim.spacing_max_nm !== undefined) {
        hintS.textContent = `d = ${lim.spacing_nm?.toFixed(2)} нм; дозволено ${lim.spacing_min_nm}…${lim.spacing_max_nm.toFixed(2)} нм`;
    }
    const warnEl = document.getElementById('validation-warnings');
    if (warnEl) {
        warnEl.innerHTML = (data.warnings || []).map((w) => `<div>⚠ ${w}</div>`).join('');
    }
}

function updatePreviewMeta(data) {
    const v = readVisualParams();
    const k = data.wave_vector || [0, 0, 1];
    const epsStr = `ε(${data.preview_wavelength_nm?.toFixed(0)}нм)=${data.epsilon_real?.toFixed(1)}${data.epsilon_imag >= 0 ? '+' : ''}${data.epsilon_imag?.toFixed(1)}i`;
    const nfOn = v.show_near_field ? `${(data.near_field || []).length} NF-точок` : 'NF вимкнено';
    document.getElementById('preview-meta').innerHTML =
        `<span><strong>${data.material}</strong> · ${data.shape}</span>` +
        `<span>${epsStr}</span>` +
        `<span>|α|=${data.polarizability_abs_um3?.toExponential(2)} μm³</span>` +
        `<span>відгук |p|/|αE₀|=${data.response_strength?.toFixed(2)}</span>` +
        `<span>λ<sub>3D</sub>=${data.preview_wavelength_nm?.toFixed(0)} нм</span>` +
        `<span>${nfOn}</span>` +
        `<span>E₀∥${v.polarization}</span>`;
}

async function updatePreview() {
    const text = document.getElementById('preview-text');
    const reqId = ++previewRequestId;
    text.textContent = 'Розрахунок DDA для 3D…';
    try {
        const res = await fetch('/api/preview?' + formParams().toString() + '&_ts=' + Date.now());
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const data = await res.json();
        if (reqId !== previewRequestId) return;
        lastPreviewData = data;
        const pwEl = document.getElementById('preview_wavelength_nm');
        if (pwEl && data.preview_wavelength_nm && document.activeElement !== pwEl) {
            pwEl.value = Math.round(data.preview_wavelength_nm);
        }
        applyLimitsFromPreview(data);
        renderScene(data);
        text.textContent = `3D: ${data.material}, λ=${data.preview_wavelength_nm?.toFixed(0)} нм, ε=${data.epsilon_real?.toFixed(1)}+${data.epsilon_imag?.toFixed(1)}i, відгук=${data.response_strength?.toFixed(2)}`;
        updatePreviewMeta(data);
    } catch (e) {
        if (reqId === previewRequestId) text.textContent = 'Помилка: ' + e.message;
    }
}

function schedulePreviewUpdate() {
    if (previewTimer) clearTimeout(previewTimer);
    previewTimer = setTimeout(updatePreview, 320);
}

function toggleAmbientIndex() {
    const env = document.getElementById('environment')?.value;
    const ambient = document.getElementById('ambient_index');
    if (!ambient) return;
    if (env === 'Custom') ambient.removeAttribute('readonly');
    else {
        const values = { Air: 1.0, Water: 1.33, Glass: 1.5 };
        ambient.value = values[env] || 1.0;
        ambient.setAttribute('readonly', 'readonly');
    }
    schedulePreviewUpdate();
}
