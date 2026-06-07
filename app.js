/**
 * FaceLog — Face Recognition Attendance System
 *
 * Storage: Firebase Firestore (free tier, cross-device)
 * Face ML: face-api.js / @vladmandic (runs entirely in browser)
 *
 * Collections:
 *   users/      — { id, name, email, department, photo, descriptors[], registeredAt }
 *   attendance/ — { userId, name, idNumber, department, photo, confidence, timestamp, date, time }
 */

// ─── Constants ───────────────────────────────────────────────────────────────
const MODELS_URL    = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model';
const MATCH_THRESH  = 0.50;
const CAPTURE_SAMPLES = 3;
const FB_CONFIG_KEY = 'facelog_firebase_config';

// ─── State ───────────────────────────────────────────────────────────────────
let firestore       = null;   // Firestore instance
let faceMatcher     = null;
let loginStream     = null;
let regStream       = null;
let loginScanActive = false;
let capturedFaceDescriptor = null;
let capturedPhotoB64       = null;

// Firebase SDK globals (loaded from CDN)
let fbApp, fbFirestore, fbCollection, fbDoc, fbAddDoc,
    fbSetDoc, fbGetDocs, fbDeleteDoc, fbQuery, fbWhere, fbOrderBy;

// ─── Firebase Config UI ──────────────────────────────────────────────────────
function showConfigScreen() {
  document.getElementById('loading-screen').classList.add('hidden');
  document.getElementById('config-screen').classList.remove('hidden');
}

function hideConfigScreen() {
  document.getElementById('config-screen').classList.add('hidden');
}

document.getElementById('btn-save-config').addEventListener('click', () => {
  const raw = document.getElementById('firebase-config-input').value.trim();
  if (!raw) { showConfigError('Paste your Firebase config object above.'); return; }

  // Accept either a plain JS object literal or JSON
  let cfg;
  try {
    // Try JSON first
    cfg = JSON.parse(raw);
  } catch (_) {
    try {
      // Evaluate as JS object literal (safe — no network, no DOM)
      // eslint-disable-next-line no-new-func
      cfg = Function('"use strict"; return (' + raw + ')')();
    } catch (e) {
      showConfigError('Could not parse config. Make sure you copied the full firebaseConfig object. ' + e.message);
      return;
    }
  }

  const required = ['apiKey','authDomain','projectId'];
  const missing  = required.filter(k => !cfg[k]);
  if (missing.length) {
    showConfigError('Missing fields: ' + missing.join(', ') + '. Copy the full config from Firebase console.');
    return;
  }

  localStorage.setItem(FB_CONFIG_KEY, JSON.stringify(cfg));
  hideConfigScreen();
  document.getElementById('loading-screen').classList.remove('hidden');
  initFirebaseAndBoot(cfg);
});

document.getElementById('btn-reset-config').addEventListener('click', () => {
  if (!confirm('Clear Firebase config and start over?')) return;
  localStorage.removeItem(FB_CONFIG_KEY);
  location.reload();
});

function showConfigError(msg) {
  const el = document.getElementById('config-error');
  el.textContent = msg;
  el.classList.remove('hidden');
}

// ─── Firebase Init ───────────────────────────────────────────────────────────
async function initFirebaseAndBoot(cfg) {
  const loadMsg = document.getElementById('load-msg');
  try {
    loadMsg.textContent = 'Connecting to Firebase…';

    const { initializeApp }             = await import('https://www.gstatic.com/firebasejs/10.12.2/firebase-app.js');
    const { getFirestore, collection, doc, addDoc,
            setDoc, getDocs, deleteDoc,
            query, where, orderBy }     = await import('https://www.gstatic.com/firebasejs/10.12.2/firebase-firestore.js');

    fbCollection = collection;
    fbDoc        = doc;
    fbAddDoc     = addDoc;
    fbSetDoc     = setDoc;
    fbGetDocs    = getDocs;
    fbDeleteDoc  = deleteDoc;
    fbQuery      = query;
    fbWhere      = where;
    fbOrderBy    = orderBy;

    fbApp       = initializeApp(cfg);
    firestore   = getFirestore(fbApp);

    loadMsg.textContent = 'Loading face detection model…';
    await faceapi.nets.ssdMobilenetv1.loadFromUri(MODELS_URL);

    loadMsg.textContent = 'Loading face landmarks model…';
    await faceapi.nets.faceLandmark68Net.loadFromUri(MODELS_URL);

    loadMsg.textContent = 'Loading face recognition model…';
    await faceapi.nets.faceRecognitionNet.loadFromUri(MODELS_URL);

    loadMsg.textContent = 'Loading registered users…';
    await rebuildFaceMatcher();

    document.getElementById('loading-screen').classList.add('hidden');
    document.getElementById('app').classList.remove('hidden');

    const users = await getAllUsers();
    setStatus('login-status', `Ready — ${users.length} user(s) in cloud. Click "Start Camera".`);
    setStatus('reg-status',   'Ready. Fill in details and start the camera.');
    if (users.length > 0) toast(`${users.length} user(s) loaded from cloud.`, 'ok', 4000);

  } catch (err) {
    console.error(err);
    loadMsg.textContent = `Error: ${err.message}`;
    loadMsg.style.color = '#ff4444';
    // If it looks like a bad config, let them fix it
    if (err.message.includes('invalid-api-key') || err.message.includes('app/invalid')) {
      setTimeout(() => {
        localStorage.removeItem(FB_CONFIG_KEY);
        document.getElementById('loading-screen').classList.add('hidden');
        showConfigScreen();
        showConfigError('Invalid API key or config. Please check and re-enter.');
      }, 1500);
    }
  }
}

// ─── Firestore helpers ───────────────────────────────────────────────────────
function usersCol()      { return fbCollection(firestore, 'users'); }
function attendanceCol() { return fbCollection(firestore, 'attendance'); }

async function getAllUsers() {
  const snap = await fbGetDocs(usersCol());
  return snap.docs.map(d => ({ ...d.data(), _docId: d.id }));
}

async function getAllAttendance() {
  const snap = await fbGetDocs(attendanceCol());
  return snap.docs.map(d => ({ ...d.data(), _docId: d.id }));
}

async function saveUser(user) {
  // Use user.id as document ID so updates are idempotent
  await fbSetDoc(fbDoc(firestore, 'users', user.id), user);
}

async function deleteUser(userId) {
  await fbDeleteDoc(fbDoc(firestore, 'users', userId));
}

async function addAttendance(record) {
  await fbAddDoc(attendanceCol(), record);
}

async function deleteAttendanceForUser(userId) {
  const snap = await fbGetDocs(
    fbQuery(attendanceCol(), fbWhere('userId', '==', userId))
  );
  const dels = snap.docs.map(d => fbDeleteDoc(d.ref));
  await Promise.all(dels);
}

async function clearTodayAttendance(today) {
  const snap = await fbGetDocs(
    fbQuery(attendanceCol(), fbWhere('date', '==', today))
  );
  const dels = snap.docs.map(d => fbDeleteDoc(d.ref));
  await Promise.all(dels);
}

// ─── Face descriptor serialization ───────────────────────────────────────────
function descriptorToArray(d) { return Array.from(d); }
function arrayToDescriptor(a) { return new Float32Array(a); }

// ─── Build face matcher ───────────────────────────────────────────────────────
async function rebuildFaceMatcher() {
  const users = await getAllUsers();
  if (users.length === 0) { faceMatcher = null; return; }
  const labeled = users.map(u => {
    const descs = u.descriptors.map(arrayToDescriptor);
    return new faceapi.LabeledFaceDescriptors(u.id, descs);
  });
  faceMatcher = new faceapi.FaceMatcher(labeled, MATCH_THRESH);
}

// ─── Toast ───────────────────────────────────────────────────────────────────
let toastTimer = null;
function toast(msg, type = '', duration = 3000) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className   = `show ${type}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.className = ''; }, duration);
}

// ─── Status box ──────────────────────────────────────────────────────────────
function setStatus(elId, msg, type = '') {
  const el = document.getElementById(elId);
  if (!el) return;
  el.textContent = msg;
  el.className   = `status-box ${type}`;
}

// ─── Camera helpers ──────────────────────────────────────────────────────────
async function startCamera(videoEl) {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user', frameRate: { ideal: 15 } }
  });
  videoEl.srcObject = stream;
  await new Promise(r => { videoEl.onloadedmetadata = r; });
  await videoEl.play();
  return stream;
}

function stopStream(stream) {
  if (stream) stream.getTracks().forEach(t => t.stop());
}

// ─── Face detection ───────────────────────────────────────────────────────────
async function detectFace(videoEl) {
  return faceapi
    .detectSingleFace(videoEl, new faceapi.SsdMobilenetv1Options({ minConfidence: 0.4 }))
    .withFaceLandmarks()
    .withFaceDescriptor();
}

async function captureSampledDescriptor(videoEl, statusId, samples = CAPTURE_SAMPLES) {
  const descriptors = [];
  setStatus(statusId, `Sampling face (0 / ${samples})…`, 'warn');
  for (let i = 0; i < samples; i++) {
    await new Promise(r => setTimeout(r, 400));
    const det = await detectFace(videoEl);
    if (!det) {
      setStatus(statusId, `Sample ${i + 1} failed — keep face centred.`, 'err');
      await new Promise(r => setTimeout(r, 600));
      i--; continue;
    }
    descriptors.push(det.descriptor);
    setStatus(statusId, `Sampling face (${descriptors.length} / ${samples})…`, 'warn');
  }
  const avg = new Float32Array(128);
  for (const d of descriptors) for (let j = 0; j < 128; j++) avg[j] += d[j];
  for (let j = 0; j < 128; j++) avg[j] /= samples;
  return avg;
}

function snapPhoto(videoEl) {
  const c = document.createElement('canvas');
  c.width  = videoEl.videoWidth  || 320;
  c.height = videoEl.videoHeight || 240;
  c.getContext('2d').drawImage(videoEl, 0, 0);
  return c.toDataURL('image/jpeg', 0.7);
}

function drawDetectionBox(overlayCanvas, videoEl, detection) {
  const dims = faceapi.matchDimensions(overlayCanvas, videoEl, true);
  faceapi.draw.drawDetections(overlayCanvas, faceapi.resizeResults(detection, dims));
}

function clearOverlay(canvas) {
  if (canvas) canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
}

// ─── Date / time ─────────────────────────────────────────────────────────────
function nowDate() { return new Date().toLocaleDateString('en-CA'); }
function nowTime() { return new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' }); }
function fmtDateTime(isoString) {
  const d = new Date(isoString);
  return {
    date: d.toLocaleDateString('en-CA'),
    time: d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  };
}

// ─────────────────────────────────────────────────────────────────────────────
//  TAB NAVIGATION
// ─────────────────────────────────────────────────────────────────────────────
document.querySelectorAll('.nav-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.tab;
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(t => { t.classList.remove('active'); t.classList.add('hidden'); });
    btn.classList.add('active');
    const target = document.getElementById(`tab-${tab}`);
    target.classList.remove('hidden');
    target.classList.add('active');

    if (tab !== 'login')    { stopStream(loginStream); loginStream = null; loginScanActive = false; }
    if (tab !== 'register') resetRegCapture();
    if (tab === 'records')  loadRecordsTab();
  });
});

// ─────────────────────────────────────────────────────────────────────────────
//  INNER TABS (Records)
// ─────────────────────────────────────────────────────────────────────────────
document.querySelectorAll('.inner-tab').forEach(btn => {
  btn.addEventListener('click', () => {
    const which = btn.dataset.inner;
    document.querySelectorAll('.inner-tab').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.inner-panel').forEach(p => p.classList.add('hidden'));
    btn.classList.add('active');
    document.getElementById(`inner-${which}`).classList.remove('hidden');
  });
});

// ─────────────────────────────────────────────────────────────────────────────
//  LOGIN TAB
// ─────────────────────────────────────────────────────────────────────────────
const loginVideo   = document.getElementById('login-video');
const loginOverlay = document.getElementById('login-overlay');
const loginCamWrap = document.getElementById('login-cam-wrap');

document.getElementById('btn-start-login').addEventListener('click', async () => {
  try {
    setStatus('login-status', 'Requesting camera…', 'warn');
    loginStream = await startCamera(loginVideo);
    document.getElementById('btn-start-login').disabled = true;
    document.getElementById('btn-scan-login').disabled  = false;
    setStatus('login-status', 'Camera ready. Click "Scan Face" and look directly at the camera.', 'ok');
  } catch (err) {
    setStatus('login-status', `Camera error: ${err.message}`, 'err');
  }
});

document.getElementById('btn-scan-login').addEventListener('click', async () => {
  if (loginScanActive) return;
  if (!faceMatcher) {
    setStatus('login-status', 'No users registered yet. Go to Register tab first.', 'err');
    return;
  }

  loginScanActive = true;
  document.getElementById('btn-scan-login').disabled = true;
  document.getElementById('login-result').classList.add('hidden');
  loginCamWrap.classList.add('scanning');

  try {
    setStatus('login-status', 'Detecting face — hold still…');
    clearOverlay(loginOverlay);

    let detection = null;
    for (let attempt = 0; attempt < 5; attempt++) {
      await new Promise(r => setTimeout(r, 500));
      detection = await detectFace(loginVideo);
      if (detection) break;
      setStatus('login-status', `Attempt ${attempt + 1}/5 — no face detected…`, 'warn');
    }

    if (!detection) {
      setStatus('login-status', 'No face detected. Check lighting and face the camera squarely.', 'err');
      loginCamWrap.classList.remove('scanning');
      document.getElementById('btn-scan-login').disabled = false;
      loginScanActive = false;
      return;
    }

    drawDetectionBox(loginOverlay, loginVideo, detection);
    const match = faceMatcher.findBestMatch(detection.descriptor);

    if (match.label === 'unknown') {
      setStatus('login-status', `Face not recognised (score: ${match.distance.toFixed(3)}). Not registered.`, 'err');
      loginCamWrap.classList.remove('scanning');
      document.getElementById('btn-scan-login').disabled = false;
      loginScanActive = false;
      return;
    }

    const confidence = Math.round((1 - match.distance) * 100);
    const users      = await getAllUsers();
    const user       = users.find(u => u.id === match.label);

    if (!user) {
      setStatus('login-status', 'Match found but user record missing. Please re-register.', 'err');
      loginCamWrap.classList.remove('scanning');
      document.getElementById('btn-scan-login').disabled = false;
      loginScanActive = false;
      return;
    }

    // Check duplicate attendance today
    const today    = nowDate();
    const allAtt   = await getAllAttendance();
    const already  = allAtt.find(a => a.userId === user.id && a.date === today);

    if (already) {
      setStatus('login-status', `${user.name} already marked attended today (${already.time}).`, 'warn');
      showLoginResult(user, confidence, already.timestamp, 'ALREADY ATTENDED');
      loginCamWrap.classList.remove('scanning');
      document.getElementById('btn-scan-login').disabled = false;
      loginScanActive = false;
      return;
    }

    const ts = new Date().toISOString();
    await addAttendance({
      userId:     user.id,
      name:       user.name,
      idNumber:   user.id,
      department: user.department || '',
      photo:      user.photo      || '',
      confidence,
      timestamp:  ts,
      date:       today,
      time:       nowTime()
    });

    setStatus('login-status', `✓ ${user.name} — attendance marked!`, 'ok');
    toast(`Attendance marked for ${user.name}`, 'ok');
    showLoginResult(user, confidence, ts, 'ATTENDED');

  } catch (err) {
    console.error(err);
    setStatus('login-status', `Error: ${err.message}`, 'err');
  }

  loginCamWrap.classList.remove('scanning');
  document.getElementById('btn-scan-login').disabled = false;
  loginScanActive = false;
});

function showLoginResult(user, confidence, timestamp, badgeText) {
  const result = document.getElementById('login-result');
  result.classList.remove('hidden');
  document.getElementById('login-result-photo').src  = user.photo || '';
  document.getElementById('result-name').textContent  = user.name;
  document.getElementById('result-meta').textContent  = `${user.id}${user.department ? '  ·  ' + user.department : ''}`;
  const { date, time } = fmtDateTime(timestamp);
  document.getElementById('result-time').textContent  = `${date}  ${time}`;
  document.getElementById('result-badge').textContent = badgeText;
  document.getElementById('result-badge').style.color = badgeText === 'ATTENDED' ? 'var(--success)' : 'var(--warn)';
}

// ─────────────────────────────────────────────────────────────────────────────
//  REGISTER TAB
// ─────────────────────────────────────────────────────────────────────────────
const regVideo   = document.getElementById('reg-video');
const regOverlay = document.getElementById('reg-overlay');
const regCamWrap = document.getElementById('reg-cam-wrap');

document.getElementById('btn-start-reg').addEventListener('click', async () => {
  stopStream(regStream);
  regStream = null;
  regVideo.srcObject = null;
  document.getElementById('btn-start-reg').disabled = true;
  document.getElementById('btn-capture').disabled   = true;

  try {
    setStatus('reg-status', 'Requesting camera…', 'warn');
    regStream = await startCamera(regVideo);
    document.getElementById('btn-capture').disabled = false;
    setStatus('reg-status', 'Camera ready. Position your face, then click "Capture Photo".', 'ok');
  } catch (err) {
    document.getElementById('btn-start-reg').disabled = false;
    setStatus('reg-status', `Camera error: ${err.message}`, 'err');
  }
});

document.getElementById('btn-capture').addEventListener('click', async () => {
  document.getElementById('btn-capture').disabled = true;
  regCamWrap.classList.add('scanning');
  setStatus('reg-status', 'Detecting face — look directly at the camera…', 'warn');
  clearOverlay(regOverlay);
  capturedFaceDescriptor = null;
  capturedPhotoB64       = null;

  try {
    let detection = null;
    for (let attempt = 0; attempt < 5; attempt++) {
      await new Promise(r => setTimeout(r, 500));
      detection = await detectFace(regVideo);
      if (detection) break;
      setStatus('reg-status', `Attempt ${attempt + 1}/5 — no face found. Check lighting.`, 'warn');
    }

    if (!detection) {
      setStatus('reg-status', 'Could not detect a face. Move closer, improve lighting, try again.', 'err');
      regCamWrap.classList.remove('scanning');
      document.getElementById('btn-capture').disabled = false;
      return;
    }

    drawDetectionBox(regOverlay, regVideo, detection);
    capturedPhotoB64       = snapPhoto(regVideo);
    capturedFaceDescriptor = await captureSampledDescriptor(regVideo, 'reg-status', CAPTURE_SAMPLES);

    document.getElementById('reg-preview-img').src = capturedPhotoB64;
    document.getElementById('reg-preview').classList.remove('hidden');
    setStatus('reg-status', 'Face captured! Fill in your details and click "Register".', 'ok');
    document.getElementById('btn-register').disabled = false;
    document.getElementById('btn-capture').disabled  = false;

  } catch (err) {
    console.error(err);
    setStatus('reg-status', `Capture error: ${err.message}`, 'err');
    document.getElementById('btn-capture').disabled = false;
  }

  regCamWrap.classList.remove('scanning');
});

document.getElementById('btn-register').addEventListener('click', async () => {
  const name  = document.getElementById('reg-name').value.trim();
  const id    = document.getElementById('reg-id').value.trim();
  const email = document.getElementById('reg-email').value.trim();
  const dept  = document.getElementById('reg-dept').value.trim();

  if (!name || !id)             { toast('Full Name and ID are required.', 'err'); return; }
  if (!capturedFaceDescriptor)  { toast('Please capture your face first.', 'err'); return; }

  document.getElementById('btn-register').disabled = true;
  setStatus('reg-status', 'Saving to cloud…', 'warn');

  try {
    const users = await getAllUsers();
    if (users.find(u => u.id === id)) {
      toast(`ID "${id}" is already registered.`, 'err');
      document.getElementById('btn-register').disabled = false;
      return;
    }

    await saveUser({
      id, name, email, department: dept,
      photo:       capturedPhotoB64,
      descriptors: [descriptorToArray(capturedFaceDescriptor)],
      registeredAt: new Date().toISOString()
    });

    await rebuildFaceMatcher();
    toast(`${name} registered successfully!`, 'ok');
    setStatus('reg-status', `✓ ${name} saved to cloud. Click "Start Camera" to register another.`, 'ok');

    // Stop camera, reset everything for next registration
    stopStream(regStream);
    regStream = null;
    regVideo.srcObject = null;

    document.getElementById('reg-name').value  = '';
    document.getElementById('reg-id').value    = '';
    document.getElementById('reg-email').value = '';
    document.getElementById('reg-dept').value  = '';
    capturedFaceDescriptor = null;
    capturedPhotoB64       = null;
    clearOverlay(regOverlay);
    document.getElementById('reg-preview').classList.add('hidden');
    document.getElementById('btn-register').disabled  = true;
    document.getElementById('btn-capture').disabled   = true;
    document.getElementById('btn-start-reg').disabled = false;

  } catch (err) {
    console.error(err);
    setStatus('reg-status', `Save failed: ${err.message}`, 'err');
    document.getElementById('btn-register').disabled = false;
  }
});

function resetRegCapture() {
  stopStream(regStream);
  regStream = null;
  regVideo.srcObject = null;
  capturedFaceDescriptor = null;
  capturedPhotoB64       = null;
  document.getElementById('reg-preview').classList.add('hidden');
  document.getElementById('btn-capture').disabled   = true;
  document.getElementById('btn-register').disabled  = true;
  document.getElementById('btn-start-reg').disabled = false;
  clearOverlay(regOverlay);
}

// ─────────────────────────────────────────────────────────────────────────────
//  RECORDS TAB
// ─────────────────────────────────────────────────────────────────────────────
let allAttendanceCache = [];
let allUsersCache      = [];

async function loadRecordsTab() {
  await loadAttendanceTable();
  await loadUsersTable();
}

async function loadAttendanceTable(filter = '') {
  allAttendanceCache = await getAllAttendance();
  allAttendanceCache.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));

  const tbody = document.getElementById('attendance-tbody');
  const empty = document.getElementById('attendance-empty');
  tbody.innerHTML = '';

  const filtered = filter
    ? allAttendanceCache.filter(r =>
        r.name.toLowerCase().includes(filter) ||
        (r.idNumber || '').toLowerCase().includes(filter))
    : allAttendanceCache;

  if (filtered.length === 0) {
    empty.classList.remove('hidden');
  } else {
    empty.classList.add('hidden');
    for (const rec of filtered) {
      const { date, time } = fmtDateTime(rec.timestamp);
      const conf = rec.confidence ?? 0;
      const tr   = document.createElement('tr');
      tr.innerHTML = `
        <td><img class="table-photo" src="${rec.photo || ''}" alt="" /></td>
        <td>${rec.name}</td>
        <td><code>${rec.idNumber || '—'}</code></td>
        <td>${rec.department || '—'}</td>
        <td>${date}</td>
        <td>${time}</td>
        <td>
          <div class="conf-bar">
            <div class="conf-fill" style="width:${conf}px; max-width:60px;"></div>
            <span class="conf-text">${conf}%</span>
          </div>
        </td>`;
      tbody.appendChild(tr);
    }
  }
}

async function loadUsersTable(filter = '') {
  allUsersCache = await getAllUsers();
  allUsersCache.sort((a, b) => a.name.localeCompare(b.name));

  const tbody = document.getElementById('users-tbody');
  const empty = document.getElementById('users-empty');
  tbody.innerHTML = '';

  const filtered = filter
    ? allUsersCache.filter(u =>
        u.name.toLowerCase().includes(filter) ||
        u.id.toLowerCase().includes(filter))
    : allUsersCache;

  if (filtered.length === 0) {
    empty.classList.remove('hidden');
  } else {
    empty.classList.add('hidden');
    for (const u of filtered) {
      const { date } = fmtDateTime(u.registeredAt);
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><img class="table-photo" src="${u.photo || ''}" alt="" /></td>
        <td>${u.name}</td>
        <td><code>${u.id}</code></td>
        <td>${u.email || '—'}</td>
        <td>${u.department || '—'}</td>
        <td>${date}</td>
        <td><button class="delete-btn" data-id="${u.id}">Delete</button></td>`;
      tbody.appendChild(tr);
    }

    tbody.querySelectorAll('.delete-btn').forEach(btn => {
      btn.addEventListener('click', async () => {
        const uid = btn.dataset.id;
        if (!confirm(`Delete user "${uid}" and all their attendance records?`)) return;
        try {
          await deleteUser(uid);
          await deleteAttendanceForUser(uid);
          await rebuildFaceMatcher();
          toast('User deleted from cloud.', 'warn');
          loadRecordsTab();
        } catch (err) {
          toast(`Delete failed: ${err.message}`, 'err');
        }
      });
    });
  }
}

document.getElementById('search-records').addEventListener('input', (e) => {
  const q = e.target.value.trim().toLowerCase();
  loadAttendanceTable(q);
  loadUsersTable(q);
});

document.getElementById('btn-clear-attendance').addEventListener('click', async () => {
  const today = nowDate();
  if (!confirm(`Clear all attendance records for today (${today})?`)) return;
  try {
    await clearTodayAttendance(today);
    toast("Today's attendance cleared.", 'warn');
    loadRecordsTab();
  } catch (err) {
    toast(`Failed: ${err.message}`, 'err');
  }
});

document.getElementById('btn-export').addEventListener('click', async () => {
  const all = await getAllAttendance();
  if (all.length === 0) { toast('No records to export.', 'warn'); return; }
  all.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));

  const rows = [['Name', 'ID', 'Department', 'Date', 'Time', 'Confidence %']];
  for (const r of all) {
    const { date, time } = fmtDateTime(r.timestamp);
    rows.push([r.name, r.idNumber || '', r.department || '', date, time, r.confidence ?? '']);
  }
  const csv  = rows.map(row => row.map(c => `"${String(c).replace(/"/g, '""')}"`).join(',')).join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href = url; a.download = `attendance_${nowDate()}.csv`; a.click();
  URL.revokeObjectURL(url);
  toast('CSV exported.', 'ok');
});

// ─────────────────────────────────────────────────────────────────────────────
//  BOOT
// ─────────────────────────────────────────────────────────────────────────────
// ── Hardcoded Firebase config — no paste needed on every boot ──
const HARDCODED_CONFIG = {
  apiKey:            "AIzaSyBFlbrtcd_jE2R5j65s_7NQaBVE0DIyNeU",
  authDomain:        "ect1-3bb7c.firebaseapp.com",
  projectId:         "ect1-3bb7c",
  storageBucket:     "ect1-3bb7c.firebasestorage.app",
  messagingSenderId: "936394461021",
  appId:             "1:936394461021:web:d10349ebea769358fb22b4"
};

function waitForFaceAPI() {
  if (typeof faceapi !== 'undefined') {
    // Always use hardcoded config — skip the config screen entirely
    localStorage.setItem(FB_CONFIG_KEY, JSON.stringify(HARDCODED_CONFIG));
    initFirebaseAndBoot(HARDCODED_CONFIG);
  } else {
    setTimeout(waitForFaceAPI, 100);
  }
}

waitForFaceAPI();
