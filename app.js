/**
 * FaceLog — Face Recognition Attendance System
 * 
 * Architecture:
 *  - face-api.js  → face detection + 128-d descriptor extraction
 *  - IndexedDB    → persistent storage for users + attendance (no server needed)
 *  - All face matching done in-browser with euclidean distance
 * 
 * Realistic approach for laptop webcams:
 *  - Uses SSD MobileNet (faster, works on lower-quality video)
 *  - Generous matching threshold (0.5) to handle lighting variance
 *  - Captures multiple-frame descriptors during registration for robustness
 */

// ─── Constants ───────────────────────────────────────────────────────────────
const MODELS_URL   = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model';
const MATCH_THRESH = 0.50;   // euclidean distance threshold (lower = stricter)
const CAPTURE_SAMPLES = 3;   // how many descriptor samples to average at registration

// ─── State ───────────────────────────────────────────────────────────────────
let modelsReady    = false;
let loginStream    = null;
let regStream      = null;
let capturedFaceDescriptor = null;  // averaged descriptor for the captured face
let capturedPhotoB64       = null;  // base64 photo taken at capture time
let faceMatcher    = null;          // faceapi.FaceMatcher, rebuilt whenever users change
let loginScanActive = false;

// ─── DB ──────────────────────────────────────────────────────────────────────
let db;
const DB_NAME    = 'FaceLogDB';
const DB_VERSION = 1;

function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);

    req.onupgradeneeded = (e) => {
      const database = e.target.result;
      if (!database.objectStoreNames.contains('users')) {
        const userStore = database.createObjectStore('users', { keyPath: 'id' });
        userStore.createIndex('name', 'name', { unique: false });
      }
      if (!database.objectStoreNames.contains('attendance')) {
        const attStore = database.createObjectStore('attendance', { autoIncrement: true });
        attStore.createIndex('userId', 'userId', { unique: false });
        attStore.createIndex('date',   'date',   { unique: false });
      }
    };

    req.onsuccess = (e) => { db = e.target.result; resolve(db); };
    req.onerror   = (e) => reject(e.target.error);
  });
}

function dbGetAll(storeName) {
  return new Promise((resolve, reject) => {
    const tx  = db.transaction(storeName, 'readonly');
    const req = tx.objectStore(storeName).getAll();
    req.onsuccess = () => resolve(req.result);
    req.onerror   = () => reject(req.error);
  });
}

function dbPut(storeName, data) {
  return new Promise((resolve, reject) => {
    const tx  = db.transaction(storeName, 'readwrite');
    const req = tx.objectStore(storeName).put(data);
    req.onsuccess = () => resolve(req.result);
    req.onerror   = () => reject(req.error);
  });
}

function dbDelete(storeName, key) {
  return new Promise((resolve, reject) => {
    const tx  = db.transaction(storeName, 'readwrite');
    const req = tx.objectStore(storeName).delete(key);
    req.onsuccess = () => resolve();
    req.onerror   = () => reject(req.error);
  });
}

function dbAdd(storeName, data) {
  return new Promise((resolve, reject) => {
    const tx  = db.transaction(storeName, 'readwrite');
    const req = tx.objectStore(storeName).add(data);
    req.onsuccess = () => resolve(req.result);
    req.onerror   = () => reject(req.error);
  });
}

function dbClear(storeName) {
  return new Promise((resolve, reject) => {
    const tx  = db.transaction(storeName, 'readwrite');
    const req = tx.objectStore(storeName).clear();
    req.onsuccess = () => resolve();
    req.onerror   = () => reject(req.error);
  });
}

// ─── Face descriptor serialization ───────────────────────────────────────────
// Float32Array can't be stored in IndexedDB directly → convert to/from array
function descriptorToArray(d) { return Array.from(d); }
function arrayToDescriptor(a) { return new Float32Array(a); }

// ─── Build face matcher from stored users ────────────────────────────────────
async function rebuildFaceMatcher() {
  const users = await dbGetAll('users');
  if (users.length === 0) { faceMatcher = null; return; }

  const labeledDescriptors = users.map(u => {
    const descriptors = u.descriptors.map(arrayToDescriptor);
    return new faceapi.LabeledFaceDescriptors(u.id, descriptors);
  });

  faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, MATCH_THRESH);
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

// ─── Set status box ──────────────────────────────────────────────────────────
function setStatus(elId, msg, type = '') {
  const el = document.getElementById(elId);
  el.textContent = msg;
  el.className   = `status-box ${type}`;
}

// ─── Start webcam ─────────────────────────────────────────────────────────────
async function startCamera(videoEl) {
  const constraints = {
    video: {
      width:       { ideal: 640 },
      height:      { ideal: 480 },
      facingMode:  'user',
      frameRate:   { ideal: 15 }
    }
  };
  const stream = await navigator.mediaDevices.getUserMedia(constraints);
  videoEl.srcObject = stream;
  await new Promise(r => { videoEl.onloadedmetadata = r; });
  await videoEl.play();
  return stream;
}

function stopStream(stream) {
  if (stream) stream.getTracks().forEach(t => t.stop());
}

// ─── Grab a single frame and try face detection ───────────────────────────────
async function detectFace(videoEl) {
  const detection = await faceapi
    .detectSingleFace(videoEl, new faceapi.SsdMobilenetv1Options({ minConfidence: 0.4 }))
    .withFaceLandmarks()
    .withFaceDescriptor();
  return detection;
}

// ─── Capture multiple samples and average the descriptor ─────────────────────
async function captureSampledDescriptor(videoEl, statusId, samples = CAPTURE_SAMPLES) {
  const descriptors = [];
  setStatus(statusId, `Sampling face (0 / ${samples})…`, 'warn');

  for (let i = 0; i < samples; i++) {
    await new Promise(r => setTimeout(r, 400));
    const det = await detectFace(videoEl);
    if (!det) {
      setStatus(statusId, `Couldn't detect face on sample ${i + 1}. Keep face centred.`, 'err');
      await new Promise(r => setTimeout(r, 600));
      i--; // retry
      continue;
    }
    descriptors.push(det.descriptor);
    setStatus(statusId, `Sampling face (${descriptors.length} / ${samples})…`, 'warn');
  }

  // Average the descriptors
  const avg = new Float32Array(128);
  for (const d of descriptors) for (let j = 0; j < 128; j++) avg[j] += d[j];
  for (let j = 0; j < 128; j++) avg[j] /= samples;
  return avg;
}

// ─── Snapshot a canvas frame from the video ──────────────────────────────────
function snapPhoto(videoEl) {
  const c = document.createElement('canvas');
  c.width  = videoEl.videoWidth  || 320;
  c.height = videoEl.videoHeight || 240;
  c.getContext('2d').drawImage(videoEl, 0, 0);
  return c.toDataURL('image/jpeg', 0.7);
}

// ─── Draw detection box on overlay canvas ────────────────────────────────────
function drawDetectionBox(overlayCanvas, videoEl, detection) {
  const dims = faceapi.matchDimensions(overlayCanvas, videoEl, true);
  faceapi.draw.drawDetections(overlayCanvas, faceapi.resizeResults(detection, dims));
}

function clearOverlay(canvas) {
  if (canvas) canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
}

// ─── Format date/time ────────────────────────────────────────────────────────
function nowDate() {
  return new Date().toLocaleDateString('en-CA'); // YYYY-MM-DD
}
function nowTime() {
  return new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}
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

    // Stop cameras when switching away
    if (tab !== 'login')    { stopStream(loginStream); loginStream = null; loginScanActive = false; }
    if (tab !== 'register') { stopStream(regStream);   regStream   = null; resetRegCapture(); }
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
    setStatus('login-status', 'No users registered yet. Please register first.', 'err');
    return;
  }

  loginScanActive = true;
  document.getElementById('btn-scan-login').disabled = true;
  document.getElementById('login-result').classList.add('hidden');
  loginCamWrap.classList.add('scanning');

  try {
    setStatus('login-status', 'Detecting face — hold still and face the camera…');
    clearOverlay(loginOverlay);

    // Try up to 5 times with a short pause (handles slow webcams)
    let detection = null;
    for (let attempt = 0; attempt < 5; attempt++) {
      await new Promise(r => setTimeout(r, 500));
      detection = await detectFace(loginVideo);
      if (detection) break;
      setStatus('login-status', `Attempt ${attempt + 1}/5 — no face detected, adjusting…`, 'warn');
    }

    if (!detection) {
      setStatus('login-status', 'Could not detect a face. Ensure good lighting and face the camera squarely.', 'err');
      loginCamWrap.classList.remove('scanning');
      document.getElementById('btn-scan-login').disabled  = false;
      loginScanActive = false;
      return;
    }

    drawDetectionBox(loginOverlay, loginVideo, detection);

    // Match against registered users
    const match = faceMatcher.findBestMatch(detection.descriptor);

    if (match.label === 'unknown') {
      setStatus('login-status', `Face not recognised (distance: ${match.distance.toFixed(3)}). Not in the system.`, 'err');
      loginCamWrap.classList.remove('scanning');
      document.getElementById('btn-scan-login').disabled  = false;
      loginScanActive = false;
      return;
    }

    const confidence = Math.round((1 - match.distance) * 100);

    // Load user record
    const users   = await dbGetAll('users');
    const user    = users.find(u => u.id === match.label);
    if (!user) {
      setStatus('login-status', 'Match found but user record missing. Please re-register.', 'err');
      loginCamWrap.classList.remove('scanning');
      document.getElementById('btn-scan-login').disabled  = false;
      loginScanActive = false;
      return;
    }

    // Check if already attended today
    const attendanceAll = await dbGetAll('attendance');
    const today         = nowDate();
    const alreadyToday  = attendanceAll.find(a => a.userId === user.id && a.date === today);

    if (alreadyToday) {
      setStatus('login-status',
        `${user.name} already marked attended today (${alreadyToday.time}).`, 'warn');
      // Still show the result card
      showLoginResult(user, confidence, alreadyToday.timestamp, 'ALREADY ATTENDED');
      loginCamWrap.classList.remove('scanning');
      document.getElementById('btn-scan-login').disabled  = false;
      loginScanActive = false;
      return;
    }

    // Record attendance
    const ts = new Date().toISOString();
    await dbAdd('attendance', {
      userId:     user.id,
      name:       user.name,
      idNumber:   user.idNumber,
      department: user.department,
      photo:      user.photo,
      confidence,
      timestamp:  ts,
      date:       today,
      time:       nowTime()
    });

    setStatus('login-status', `✓ ${user.name} matched and attendance marked!`, 'ok');
    toast(`Attendance marked for ${user.name}`, 'ok');
    showLoginResult(user, confidence, ts, 'ATTENDED');

  } catch (err) {
    console.error(err);
    setStatus('login-status', `Error during scan: ${err.message}`, 'err');
  }

  loginCamWrap.classList.remove('scanning');
  document.getElementById('btn-scan-login').disabled  = false;
  loginScanActive = false;
});

function showLoginResult(user, confidence, timestamp, badgeText) {
  const result = document.getElementById('login-result');
  result.classList.remove('hidden');

  document.getElementById('login-result-photo').src = user.photo || '';
  document.getElementById('result-name').textContent = user.name;
  document.getElementById('result-meta').textContent = `${user.idNumber}${user.department ? '  ·  ' + user.department : ''}`;

  const { date, time } = fmtDateTime(timestamp);
  document.getElementById('result-time').textContent = `${date}  ${time}`;
  document.getElementById('result-badge').textContent = badgeText;
  document.getElementById('result-badge').style.color  = badgeText === 'ATTENDED' ? 'var(--success)' : 'var(--warn)';
}

// ─────────────────────────────────────────────────────────────────────────────
//  REGISTER TAB
// ─────────────────────────────────────────────────────────────────────────────
const regVideo   = document.getElementById('reg-video');
const regOverlay = document.getElementById('reg-overlay');
const regCamWrap = document.getElementById('reg-cam-wrap');

document.getElementById('btn-start-reg').addEventListener('click', async () => {
  // Kill any stale stream before requesting a new one (prevents glitch on re-register)
  stopStream(regStream);
  regStream = null;
  regVideo.srcObject = null;

  document.getElementById('btn-start-reg').disabled = true;
  document.getElementById('btn-capture').disabled   = true;

  try {
    setStatus('reg-status', 'Requesting camera…', 'warn');
    regStream = await startCamera(regVideo);
    document.getElementById('btn-capture').disabled = false;
    setStatus('reg-status', 'Camera ready. Position your face in the frame, then click “Capture Photo”.', 'ok');
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
      setStatus('reg-status', 'Could not detect a face. Move closer, improve lighting, and try again.', 'err');
      regCamWrap.classList.remove('scanning');
      document.getElementById('btn-capture').disabled = false;
      return;
    }

    drawDetectionBox(regOverlay, regVideo, detection);

    // Take the photo snapshot NOW (before more sampling moves things)
    capturedPhotoB64 = snapPhoto(regVideo);

    // Now collect averaged descriptor
    capturedFaceDescriptor = await captureSampledDescriptor(regVideo, 'reg-status', CAPTURE_SAMPLES);

    // Show preview
    const preview = document.getElementById('reg-preview');
    document.getElementById('reg-preview-img').src = capturedPhotoB64;
    preview.classList.remove('hidden');

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
  const name = document.getElementById('reg-name').value.trim();
  const id   = document.getElementById('reg-id').value.trim();
  const email= document.getElementById('reg-email').value.trim();
  const dept = document.getElementById('reg-dept').value.trim();

  if (!name || !id) {
    toast('Full Name and ID are required.', 'err');
    return;
  }
  if (!capturedFaceDescriptor) {
    toast('Please capture your face first.', 'err');
    return;
  }

  // Check if ID already exists
  const existing = await dbGetAll('users');
  if (existing.find(u => u.id === id)) {
    toast(`ID "${id}" is already registered.`, 'err');
    return;
  }

  const user = {
    id:          id,
    name:        name,
    email:       email,
    department:  dept,
    photo:       capturedPhotoB64,
    descriptors: [descriptorToArray(capturedFaceDescriptor)],
    registeredAt: new Date().toISOString()
  };

  await dbPut('users', user);
  await rebuildFaceMatcher();

  toast(`${name} registered successfully!`, 'ok');
  setStatus('reg-status', `✓ ${name} registered. Click "Start Camera" to register another person.`, 'ok');

  // Stop the camera fully so the next person can start a clean new session
  stopStream(regStream);
  regStream = null;
  regVideo.srcObject = null;

  // Reset form fields
  document.getElementById('reg-name').value  = '';
  document.getElementById('reg-id').value    = '';
  document.getElementById('reg-email').value = '';
  document.getElementById('reg-dept').value  = '';

  // Reset capture state and all buttons
  capturedFaceDescriptor = null;
  capturedPhotoB64       = null;
  clearOverlay(regOverlay);
  document.getElementById('reg-preview').classList.add('hidden');
  document.getElementById('btn-register').disabled  = true;
  document.getElementById('btn-capture').disabled   = true;
  document.getElementById('btn-start-reg').disabled = false;
});

function resetRegCapture() {
  // Always kill any live stream first
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
async function loadRecordsTab() {
  await loadAttendanceTable();
  await loadUsersTable();
}

let allAttendance = [];
let allUsers      = [];

async function loadAttendanceTable(filter = '') {
  allAttendance = await dbGetAll('attendance');
  // Sort newest first
  allAttendance.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));

  const tbody = document.getElementById('attendance-tbody');
  const empty = document.getElementById('attendance-empty');
  tbody.innerHTML = '';

  const filtered = filter
    ? allAttendance.filter(r =>
        r.name.toLowerCase().includes(filter) ||
        (r.idNumber || '').toLowerCase().includes(filter)
      )
    : allAttendance;

  if (filtered.length === 0) {
    empty.classList.remove('hidden');
  } else {
    empty.classList.add('hidden');
    for (const rec of filtered) {
      const { date, time } = fmtDateTime(rec.timestamp);
      const conf = rec.confidence ?? 0;
      const tr = document.createElement('tr');
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
        </td>
      `;
      tbody.appendChild(tr);
    }
  }
}

async function loadUsersTable(filter = '') {
  allUsers = await dbGetAll('users');
  allUsers.sort((a, b) => a.name.localeCompare(b.name));

  const tbody = document.getElementById('users-tbody');
  const empty = document.getElementById('users-empty');
  tbody.innerHTML = '';

  const filtered = filter
    ? allUsers.filter(u =>
        u.name.toLowerCase().includes(filter) ||
        u.id.toLowerCase().includes(filter)
      )
    : allUsers;

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
        <td><button class="delete-btn" data-id="${u.id}">Delete</button></td>
      `;
      tbody.appendChild(tr);
    }

    tbody.querySelectorAll('.delete-btn').forEach(btn => {
      btn.addEventListener('click', async () => {
        const uid = btn.dataset.id;
        if (!confirm(`Delete user "${uid}" and all their attendance records?`)) return;
        await dbDelete('users', uid);

        // Delete their attendance records
        const att = await dbGetAll('attendance');
        for (const a of att) {
          // We need the key — re-query using a cursor
        }
        // Workaround: clear and re-add all except deleted
        await clearAttendanceForUser(uid);
        await rebuildFaceMatcher();
        toast('User deleted.', 'warn');
        loadRecordsTab();
      });
    });
  }
}

async function clearAttendanceForUser(userId) {
  return new Promise((resolve, reject) => {
    const tx    = db.transaction('attendance', 'readwrite');
    const store = tx.objectStore('attendance');
    const req   = store.openCursor();
    req.onsuccess = (e) => {
      const cursor = e.target.result;
      if (cursor) {
        if (cursor.value.userId === userId) cursor.delete();
        cursor.continue();
      }
    };
    tx.oncomplete = () => resolve();
    tx.onerror    = () => reject(tx.error);
  });
}

// Search filter
document.getElementById('search-records').addEventListener('input', (e) => {
  const q = e.target.value.trim().toLowerCase();
  loadAttendanceTable(q);
  loadUsersTable(q);
});

// Clear today's attendance
document.getElementById('btn-clear-attendance').addEventListener('click', async () => {
  const today = nowDate();
  if (!confirm(`Clear all attendance records for today (${today})?`)) return;

  await new Promise((resolve, reject) => {
    const tx    = db.transaction('attendance', 'readwrite');
    const store = tx.objectStore('attendance');
    const req   = store.openCursor();
    req.onsuccess = (e) => {
      const cursor = e.target.result;
      if (cursor) {
        if (cursor.value.date === today) cursor.delete();
        cursor.continue();
      }
    };
    tx.oncomplete = () => resolve();
    tx.onerror    = () => reject(tx.error);
  });

  toast("Today's attendance cleared.", 'warn');
  loadRecordsTab();
});

// Export CSV
document.getElementById('btn-export').addEventListener('click', async () => {
  const all = await dbGetAll('attendance');
  if (all.length === 0) { toast('No records to export.', 'warn'); return; }

  const rows = [['Name', 'ID', 'Department', 'Date', 'Time', 'Confidence %']];
  for (const r of all) {
    const { date, time } = fmtDateTime(r.timestamp);
    rows.push([r.name, r.idNumber || '', r.department || '', date, time, r.confidence ?? '']);
  }

  const csv = rows.map(row =>
    row.map(cell => `"${String(cell).replace(/"/g, '""')}"`).join(',')
  ).join('\n');

  const blob = new Blob([csv], { type: 'text/csv' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = `attendance_${nowDate()}.csv`;
  a.click();
  URL.revokeObjectURL(url);
  toast('CSV exported.', 'ok');
});

// ─────────────────────────────────────────────────────────────────────────────
//  BOOT — Load Models
// ─────────────────────────────────────────────────────────────────────────────
async function boot() {
  const loadMsg = document.getElementById('load-msg');

  try {
    loadMsg.textContent = 'Opening database…';
    await openDB();

    loadMsg.textContent = 'Loading face detection model…';
    await faceapi.nets.ssdMobilenetv1.loadFromUri(MODELS_URL);

    loadMsg.textContent = 'Loading face landmarks model…';
    await faceapi.nets.faceLandmark68Net.loadFromUri(MODELS_URL);

    loadMsg.textContent = 'Loading face recognition model…';
    await faceapi.nets.faceRecognitionNet.loadFromUri(MODELS_URL);

    loadMsg.textContent = 'Building user index…';
    await rebuildFaceMatcher();

    modelsReady = true;

    // Hide loader, show app
    document.getElementById('loading-screen').classList.add('hidden');
    document.getElementById('app').classList.remove('hidden');

    setStatus('login-status', 'Models loaded. Click "Start Camera" to begin.');
    setStatus('reg-status',   'Models loaded. Fill in details and start the camera.');

    const userCount = (await dbGetAll('users')).length;
    if (userCount > 0) {
      toast(`${userCount} user(s) loaded from database.`, 'ok', 4000);
    }

  } catch (err) {
    console.error(err);
    loadMsg.textContent = `Error: ${err.message}. Check your internet connection and reload.`;
    loadMsg.style.color = '#ff4444';
  }
}

// Wait for face-api.js to be available
function waitForFaceAPI() {
  if (typeof faceapi !== 'undefined') {
    boot();
  } else {
    setTimeout(waitForFaceAPI, 100);
  }
}

waitForFaceAPI();
