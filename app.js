/* ===== app.js — FaceScope Face Detection ===== */

// ──────────────────────────────────────────────
// DOM References
// ──────────────────────────────────────────────
const video            = document.getElementById('video');
const overlay          = document.getElementById('overlay');
const startBtn         = document.getElementById('startBtn');
const stopBtn          = document.getElementById('stopBtn');
const snapshotBtn      = document.getElementById('snapshotBtn');
const cameraCard       = document.getElementById('cameraCard');
const cameraPlaceholder= document.getElementById('cameraPlaceholder');
const scannerOverlay   = document.getElementById('scannerOverlay');
const modelStatusBadge = document.getElementById('modelStatus');
const faceCountNum     = document.getElementById('faceCountNum');
const fpsBadgeEl       = document.getElementById('fpsBadge');
const fpsValue         = document.getElementById('fpsValue');
const avgConfidenceEl  = document.getElementById('avgConfidence');
const maxFacesEl       = document.getElementById('maxFaces');
const sessionTimeEl    = document.getElementById('sessionTime');
const modalBackdrop    = document.getElementById('modalBackdrop');
const snapshotCanvas   = document.getElementById('snapshotCanvas');
const downloadBtn      = document.getElementById('downloadBtn');
const modalClose       = document.getElementById('modalClose');
const modalClose2      = document.getElementById('modalClose2');

// ──────────────────────────────────────────────
// State
// ──────────────────────────────────────────────
let stream             = null;
let animationId        = null;
let modelsLoaded       = false;
let sessionStart       = null;
let sessionInterval    = null;
let maxFacesSeen       = 0;
let lastFrameTime      = performance.now();
let frameCount         = 0;
let fpsDisplayInterval = null;
let fpsAccumulator     = 0;

// ──────────────────────────────────────────────
// Background Particles
// ──────────────────────────────────────────────
function spawnParticles() {
  const container = document.getElementById('bgParticles');
  const colors = ['#3b82f6', '#06b6d4', '#8b5cf6', '#10b981'];
  for (let i = 0; i < 28; i++) {
    const p = document.createElement('div');
    p.className = 'particle';
    const size = Math.random() * 6 + 2;
    p.style.cssText = `
      width:${size}px; height:${size}px;
      left:${Math.random() * 100}%;
      background:${colors[Math.floor(Math.random() * colors.length)]};
      animation-duration:${Math.random() * 14 + 10}s;
      animation-delay:${Math.random() * 10}s;
    `;
    container.appendChild(p);
  }
}
spawnParticles();

// ──────────────────────────────────────────────
// Load face-api.js models from CDN
// ──────────────────────────────────────────────
const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/';

async function loadModels() {
  setStatus('loading', 'Loading Models…');
  try {
    await Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
      faceapi.nets.faceLandmark68TinyNet.loadFromUri(MODEL_URL),
    ]);
    modelsLoaded = true;
    setStatus('ready', 'Models Ready');
  } catch (err) {
    console.error('Model load failed:', err);
    setStatus('error', 'Model Load Failed');
  }
}

function setStatus(state, text) {
  const dot  = modelStatusBadge.querySelector('.badge-dot');
  const label= modelStatusBadge.querySelector('.badge-text');
  dot.className = 'badge-dot ' + state;
  label.textContent = text;
}

// ──────────────────────────────────────────────
// Camera Start / Stop
// ──────────────────────────────────────────────
startBtn.addEventListener('click', startCamera);
stopBtn.addEventListener('click', stopCamera);

async function startCamera() {
  if (!modelsLoaded) {
    setStatus('loading', 'Loading Models…');
    await loadModels();
  }
  if (!modelsLoaded) return;

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
      audio: false,
    });
    video.srcObject = stream;
    await new Promise(res => { video.onloadedmetadata = res; });
    video.play();

    // Show UI
    video.classList.add('visible');
    cameraPlaceholder.classList.add('hidden');
    cameraCard.classList.add('active');
    scannerOverlay.classList.add('active');
    startBtn.classList.add('hidden');
    stopBtn.classList.remove('hidden');
    snapshotBtn.disabled = false;

    // Resize canvas to match video
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // Start detection loop
    detectFaces();

    // Session timer
    sessionStart = Date.now();
    sessionInterval = setInterval(updateSessionTime, 1000);

    // FPS display update (every 500ms)
    fpsDisplayInterval = setInterval(() => {
      fpsValue.textContent = fpsAccumulator > 0 ? fpsAccumulator : '—';
      fpsAccumulator = 0;
    }, 1000);

    setStatus('ready', 'Camera Active');
  } catch (err) {
    console.error('Camera error:', err);
    setStatus('error', 'Camera Denied');
    alert('Could not access webcam. Please grant camera permission and try again.');
  }
}

function stopCamera() {
  if (animationId) { cancelAnimationFrame(animationId); animationId = null; }
  if (stream)      { stream.getTracks().forEach(t => t.stop()); stream = null; }
  if (sessionInterval) { clearInterval(sessionInterval); sessionInterval = null; }
  if (fpsDisplayInterval) { clearInterval(fpsDisplayInterval); fpsDisplayInterval = null; }
  window.removeEventListener('resize', resizeCanvas);

  video.srcObject = null;
  video.classList.remove('visible');
  cameraPlaceholder.classList.remove('hidden');
  cameraCard.classList.remove('active');
  scannerOverlay.classList.remove('active');
  stopBtn.classList.add('hidden');
  startBtn.classList.remove('hidden');
  snapshotBtn.disabled = true;

  // Clear canvas
  const ctx = overlay.getContext('2d');
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  // Reset stats
  faceCountNum.textContent = '0';
  fpsValue.textContent = '—';
  avgConfidenceEl.textContent = '—';
  maxFacesSeen = 0;
  maxFacesEl.textContent = '—';
  sessionTimeEl.textContent = '00:00';

  setStatus('ready', 'Models Ready');
}

// ──────────────────────────────────────────────
// Canvas sizing
// ──────────────────────────────────────────────
function resizeCanvas() {
  overlay.width  = video.videoWidth  || cameraCard.clientWidth;
  overlay.height = video.videoHeight || cameraCard.clientHeight;
}

// ──────────────────────────────────────────────
// Face Detection Loop
// ──────────────────────────────────────────────
const detectionOptions = new faceapi.TinyFaceDetectorOptions({
  inputSize: 416,
  scoreThreshold: 0.45,
});

async function detectFaces() {
  if (!stream) return;

  const now = performance.now();
  const delta = now - lastFrameTime;
  lastFrameTime = now;

  // Skip detection if video not ready
  if (video.readyState >= 2) {
    try {
      const detections = await faceapi
        .detectAllFaces(video, detectionOptions)
        .withFaceLandmarks(true);

      // Scale detections to canvas size
      const dims = { width: overlay.width, height: overlay.height };
      const scaled = faceapi.resizeResults(detections, dims);

      drawDetections(scaled);
      updateStats(scaled);
    } catch (e) {
      // silently ignore mid-stop errors
    }
  }

  // FPS counter
  frameCount++;
  fpsAccumulator++;

  animationId = requestAnimationFrame(detectFaces);
}

// ──────────────────────────────────────────────
// Draw Detections on Canvas
// ──────────────────────────────────────────────
function drawDetections(detections) {
  const ctx = overlay.getContext('2d');
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  detections.forEach((det, idx) => {
    const box   = det.detection.box;
    const score = det.detection.score;

    const x = box.x, y = box.y, w = box.width, h = box.height;
    const cx = x + w / 2;

    // ── Glow effect ──
    ctx.save();
    ctx.shadowColor = '#3b82f6';
    ctx.shadowBlur  = 24;
    ctx.strokeStyle = 'rgba(59,130,246,0.25)';
    ctx.lineWidth   = 10;
    ctx.strokeRect(x, y, w, h);
    ctx.restore();

    // ── Main bounding box ──
    ctx.save();
    ctx.strokeStyle = '#60a5fa';
    ctx.lineWidth   = 2.5;
    ctx.setLineDash([]);
    // Draw rounded rect manually
    roundRect(ctx, x, y, w, h, 10);
    ctx.stroke();
    ctx.restore();

    // ── Corner accents ──
    drawCornerAccents(ctx, x, y, w, h, '#06b6d4');

    // ── Landmark dots ──
    if (det.landmarks) {
      const positions = det.landmarks.positions;
      ctx.save();
      ctx.fillStyle = 'rgba(96,165,250,0.6)';
      positions.forEach(pt => {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 1.5, 0, Math.PI * 2);
        ctx.fill();
      });
      ctx.restore();
    }

    // ── Label background ──
    const label      = `FACE DETECTED  ${(score * 100).toFixed(0)}%`;
    const fontSize    = Math.max(12, Math.min(14, w * 0.05));
    ctx.font = `600 ${fontSize}px Inter, sans-serif`;
    const textW = ctx.measureText(label).width;
    const padX = 10, padY = 6;
    const lblX = x;
    const lblY = y - fontSize - padY * 2 - 4;

    // Pill background
    ctx.save();
    ctx.fillStyle = 'rgba(15,26,46,0.88)';
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 1.5;
    roundRect(ctx, lblX, lblY < 0 ? y + 2 : lblY, textW + padX * 2, fontSize + padY * 2, 7);
    ctx.fill(); ctx.stroke();
    ctx.restore();

    // Label text
    ctx.save();
    ctx.fillStyle = '#93c5fd';
    ctx.font = `600 ${fontSize}px Inter, sans-serif`;
    ctx.fillText(label, lblX + padX, (lblY < 0 ? y + 2 : lblY) + padY + fontSize - 2);
    ctx.restore();

    // ── Confidence arc ──
    drawConfidenceArc(ctx, x + w - 22, y + 22, 16, score);
  });
}

// Helper: rounded rectangle path
function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

// Helper: corner accent brackets
function drawCornerAccents(ctx, x, y, w, h, color) {
  const len = Math.min(20, w * 0.15, h * 0.15);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth   = 3;
  ctx.lineCap     = 'round';
  const pairs = [
    [[x, y + len], [x, y], [x + len, y]],
    [[x + w - len, y], [x + w, y], [x + w, y + len]],
    [[x, y + h - len], [x, y + h], [x + len, y + h]],
    [[x + w - len, y + h], [x + w, y + h], [x + w, y + h - len]],
  ];
  pairs.forEach(([a, b, c]) => {
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.lineTo(c[0], c[1]);
    ctx.stroke();
  });
  ctx.restore();
}

// Helper: small confidence arc indicator
function drawConfidenceArc(ctx, cx, cy, r, score) {
  ctx.save();
  // Track
  ctx.beginPath();
  ctx.arc(cx, cy, r, -Math.PI / 2, Math.PI * 1.5);
  ctx.strokeStyle = 'rgba(255,255,255,0.1)';
  ctx.lineWidth = 3;
  ctx.stroke();
  // Fill
  ctx.beginPath();
  ctx.arc(cx, cy, r, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * score);
  ctx.strokeStyle = score > 0.85 ? '#10b981' : score > 0.65 ? '#f59e0b' : '#ef4444';
  ctx.lineWidth = 3;
  ctx.lineCap = 'round';
  ctx.stroke();
  ctx.restore();
}

// ──────────────────────────────────────────────
// Stats Update
// ──────────────────────────────────────────────
function updateStats(detections) {
  const count = detections.length;
  faceCountNum.textContent = count;

  if (count > maxFacesSeen) {
    maxFacesSeen = count;
    maxFacesEl.textContent = maxFacesSeen;
  }

  if (count > 0) {
    const avg = detections.reduce((s, d) => s + d.detection.score, 0) / count;
    avgConfidenceEl.textContent = (avg * 100).toFixed(1) + '%';
  } else {
    avgConfidenceEl.textContent = '—';
  }
}

function updateSessionTime() {
  const elapsed = Math.floor((Date.now() - sessionStart) / 1000);
  const m = String(Math.floor(elapsed / 60)).padStart(2, '0');
  const s = String(elapsed % 60).padStart(2, '0');
  sessionTimeEl.textContent = `${m}:${s}`;
}

// ──────────────────────────────────────────────
// Snapshot
// ──────────────────────────────────────────────
snapshotBtn.addEventListener('click', takeSnapshot);

function takeSnapshot() {
  const w = overlay.width, h = overlay.height;
  snapshotCanvas.width  = w;
  snapshotCanvas.height = h;
  const sCtx = snapshotCanvas.getContext('2d');

  // Draw mirrored video frame
  sCtx.save();
  sCtx.translate(w, 0);
  sCtx.scale(-1, 1);
  sCtx.drawImage(video, 0, 0, w, h);
  sCtx.restore();

  // Draw detections (already mirrored on overlay canvas, draw it as-is but flip back)
  sCtx.save();
  sCtx.translate(w, 0);
  sCtx.scale(-1, 1);
  sCtx.drawImage(overlay, 0, 0, w, h);
  sCtx.restore();

  // Set download URL
  downloadBtn.href = snapshotCanvas.toDataURL('image/png');

  modalBackdrop.classList.remove('hidden');
}

modalClose.addEventListener('click',  () => modalBackdrop.classList.add('hidden'));
modalClose2.addEventListener('click', () => modalBackdrop.classList.add('hidden'));
modalBackdrop.addEventListener('click', (e) => {
  if (e.target === modalBackdrop) modalBackdrop.classList.add('hidden');
});

// ──────────────────────────────────────────────
// Init
// ──────────────────────────────────────────────
loadModels();
