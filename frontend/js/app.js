/* ==========================================================================
   AgriEdge Intelligence OS — Client Application Logic
   Handles Multilingual UI, Leaf Diagnostics, Schedulers, and OS Visualizations
   ========================================================================== */

let currentLang = 'en';
let i18nDict = {};
let selectedFile = null;
let isOnline = false;

// Helper utilities for null-safe DOM manipulation
function setElemText(id, text) {
  const el = document.getElementById(id);
  if (el) el.innerText = text;
}

function setElemHtml(id, html) {
  const el = document.getElementById(id);
  if (el) el.innerHTML = html;
}

function setElemStyle(id, prop, val) {
  const el = document.getElementById(id);
  if (el) el.style[prop] = val;
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
  loadDashboardData();
  searchLiveWeather();
  loadUserTaskQueue();
  loadRealResources();
  loadEvaluationMetrics();
  changeLanguage('en');
  triggerSchedulerRun();
});

// --- Section Navigation ---
function showSection(sectionId, event) {
  if (event && event.preventDefault) {
    event.preventDefault();
  }

  document.querySelectorAll('.page-section').forEach(sec => sec.classList.remove('active'));
  document.querySelectorAll('.sidebar-menu li').forEach(item => item.classList.remove('active'));

  const targetSec = document.getElementById('section-' + sectionId);
  if (targetSec) {
    targetSec.classList.add('active');
  }

  // Highlight menu item
  const activeLink = document.querySelector('.sidebar-menu a[href="#section-' + sectionId + '"]');
  if (activeLink && activeLink.parentElement) {
    activeLink.parentElement.classList.add('active');
  }

  // Update header title
  const titleMap = {
    'dashboard': 'Dashboard Overview',
    'detection': 'Crop Disease Detection',
    'assistant': 'Farmer Assistance Panel',
    'scheduler': 'Resource Scheduler',
    'resources': 'Farm Conditions & Device Status',
    'comparison': 'Scheduler Comparison',
    'files': 'File Storage Management',
    'technical': 'Technical Details'
  };
  setElemText('current-section-title', titleMap[sectionId] || 'Dashboard Overview');
  window.scrollTo(0, 0);
  return false;
}

// --- Indian Agricultural Region Selector ---
async function changeIndianRegion(regionId) {
  try {
    const res = await fetch('/api/weather/select_region/' + regionId, { method: 'POST' });
    const data = await res.json();
    if (data.success) {
      const reg = data.region;
      setElemText('dash-weather-temp', reg.temperature + ' °C');
      setElemText('dash-weather-humidity', reg.humidity + ' %');
      setElemText('dash-weather-rain', reg.rain_probability + ' %');
      setElemText('dash-weather-risk', reg.fungal_disease_risk + ' RISK');
      
      const isHigh = (reg.fungal_disease_risk === 'HIGH' || reg.fungal_disease_risk === 'SEVERE');
      setElemStyle('dash-weather-risk', 'color', isHigh ? '#d32f2f' : '#2e7d32');
      setElemText('weather-impact-desc', reg.impact);
      
      triggerSchedulerRun();
    }
  } catch (err) {
    console.error('Error selecting region:', err);
  }
}

// --- Multilingual Translation Engine ---
async function changeLanguage(langCode) {
  currentLang = langCode;
  try {
    const res = await fetch('/api/translate/' + langCode);
    i18nDict = await res.json();
    applyTranslations();
  } catch (err) {
    console.error('Translation load error:', err);
  }
}

function applyTranslations() {
  document.querySelectorAll('[data-i18n]').forEach(elem => {
    const key = elem.getAttribute('data-i18n');
    if (i18nDict[key]) {
      if (elem.tagName === 'INPUT' && elem.hasAttribute('placeholder')) {
        elem.placeholder = i18nDict[key];
      } else {
        elem.innerText = i18nDict[key];
      }
    }
  });
}

// --- Data Fetching & Dashboard Load ---
async function loadDashboardData() {
  try {
    const res = await fetch('/api/dashboard_summary');
    const data = await res.json();

    if (data.summary) {
      setElemText('stat-total', data.summary.total_analyzed);
      setElemText('stat-healthy', data.summary.healthy_count);
      setElemText('stat-diseased', data.summary.diseased_count);
      setElemText('stat-critical', data.summary.critical_count);
      setElemText('stat-confidence', data.summary.avg_confidence + '%');
    }

    if (data.weather) {
      setElemText('dash-weather-temp', data.weather.temperature + ' °C');
      setElemText('dash-weather-humidity', data.weather.humidity + ' %');
      setElemText('dash-weather-rain', data.weather.rain_probability + ' %');
      setElemText('dash-weather-risk', data.weather.fungal_disease_risk + ' RISK');
      const isHigh = (data.weather.fungal_disease_risk === 'HIGH' || data.weather.fungal_disease_risk === 'SEVERE');
      setElemStyle('dash-weather-risk', 'color', isHigh ? '#d32f2f' : '#2e7d32');
    }

    renderRecentScans(data.recent_scans);
  } catch (err) {
    console.error('Error loading dashboard summary:', err);
  }
}

function renderRecentScans(scans) {
  const tbody = document.getElementById('recent-scans-tbody');
  if (!tbody) return;
  if (!scans || scans.length === 0) {
    tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: var(--text-muted);">No scans recorded yet.</td></tr>';
    return;
  }

  tbody.innerHTML = scans.map(s => `
    <tr>
      <td>${s.timestamp}</td>
      <td><strong>${s.crop}</strong></td>
      <td>${s.disease}</td>
      <td>${s.confidence}%</td>
      <td><span class="severity-badge severity-${(s.severity || 'moderate').toLowerCase()}">${s.severity}</span></td>
    </tr>
  `).join('');
}

// --- Real Host Laptop Resources (psutil) ---
async function loadRealResources() {
  try {
    const res = await fetch('/api/resources/real');
    const data = await res.json();

    setElemText('psutil-cpu', data.cpu_utilization_pct + ' %');
    setElemText('psutil-ram', data.ram_used_mb + ' MB / ' + data.ram_total_mb + ' MB (' + data.ram_utilization_pct + '%)');
    setElemText('psutil-disk', data.storage_used_gb + ' GB / ' + data.storage_total_gb + ' GB (' + data.storage_utilization_pct + '%)');
  } catch (err) {
    console.error('Error loading real resources:', err);
  }
}

// --- Live Weather Location Search (Open-Meteo API) ---
async function searchLiveWeather() {
  const input = document.getElementById('weather-location-input');
  const city = input ? input.value.trim() : '';
  const searchCity = city || 'Thanjavur';

  try {
    const res = await fetch('/api/weather/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ location: searchCity })
    });
    const w = await res.json();

    setElemText('weather-location-name', w.location || searchCity);
    setElemText('weather-last-updated', w.last_updated || 'Just now');

    const apiBadge = document.getElementById('weather-api-badge');
    if (apiBadge) {
      apiBadge.innerText = w.is_live_api ? 'Open-Meteo Live API' : 'Last Known (Offline)';
      apiBadge.className = 'severity-badge ' + (w.is_live_api ? 'severity-none' : 'severity-high');
    }

    // Dashboard Cards
    setElemText('dash-weather-temp', w.temperature + ' °C');
    setElemText('dash-weather-humidity', w.humidity + ' %');
    setElemText('dash-weather-rain', w.rain_probability + ' %');
    setElemText('dash-weather-risk', w.fungal_disease_risk + ' RISK');
    const isHigh = (w.fungal_disease_risk === 'HIGH' || w.fungal_disease_risk === 'SEVERE');
    setElemStyle('dash-weather-risk', 'color', isHigh ? '#d32f2f' : '#2e7d32');

    // Weather Section Grid
    setElemText('res-weather-temp', w.temperature + ' °C');
    setElemText('res-weather-humidity', w.humidity + ' %');
    setElemText('res-weather-rain', w.rain_probability + ' %');
    setElemText('res-weather-risk', w.fungal_disease_risk + ' RISK');
    setElemStyle('res-weather-risk', 'color', isHigh ? '#d32f2f' : '#2e7d32');

    triggerSchedulerRun();
  } catch (err) {
    console.error('Weather search error:', err);
  }
}

// --- Leaf Image Diagnostics ---
function handleFileSelect(event) {
  const files = event.target.files;
  if (files && files[0]) {
    selectedFile = files[0];
    showPreview(selectedFile);
  }
}

function showPreview(fileOrUrl) {
  const previewBox = document.getElementById('image-preview-box');
  const previewImg = document.getElementById('preview-img');

  if (previewImg) {
    if (typeof fileOrUrl === 'string') {
      previewImg.src = fileOrUrl;
    } else {
      previewImg.src = URL.createObjectURL(fileOrUrl);
    }
  }
  if (previewBox) {
    previewBox.style.display = 'block';
  }
}

async function submitLeafImage() {
  if (!selectedFile) {
    alert('Please select or upload a leaf image file first.');
    return;
  }

  const btn = document.getElementById('analyze-btn');
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Running Inference...';
  }

  const formData = new FormData();
  formData.append('image', selectedFile);
  formData.append('model', 'efficientnet_b0');

  try {
    const res = await fetch('/api/analyze_disease', {
      method: 'POST',
      body: formData
    });
    const result = await res.json();
    renderResultCard(result);
    loadDashboardData();
  } catch (err) {
    alert('Error running disease inference: ' + err.message);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = '<i class="fa-solid fa-magnifying-glass"></i> <span data-i18n="btn_analyze">Analyze Crop Disease</span>';
    }
    applyTranslations();
  }
}

function loadSampleImage(type) {
  const canvas = document.createElement('canvas');
  canvas.width = 224;
  canvas.height = 224;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#2e7d32';
  ctx.fillRect(0, 0, 224, 224);
  ctx.fillStyle = '#81c784';
  ctx.beginPath();
  ctx.arc(112, 112, 80, 0, 2 * Math.PI);
  ctx.fill();

  canvas.toBlob(blob => {
    selectedFile = new File([blob], 'sample_leaf.jpg', { type: 'image/jpeg' });
    showPreview(selectedFile);
  });
}

function renderResultCard(res) {
  setElemText('res-crop-name', res.crop);
  setElemText('res-disease-name', res.disease);
  setElemText('res-confidence', res.confidence + ' %');
  setElemText('res-explanation', res.explanation);
  setElemText('res-model-used', res.model_name || 'EfficientNet-B0');

  const statusText = document.getElementById('res-status-text');
  if (statusText) {
    statusText.innerText = res.is_healthy ? 'HEALTHY LEAF' : 'DISEASE DETECTED';
    statusText.style.color = res.is_healthy ? '#2e7d32' : '#d32f2f';
  }

  const badge = document.getElementById('res-severity-badge');
  if (badge) {
    badge.innerText = res.severity + ' Severity';
    badge.className = 'severity-badge severity-' + (res.severity || 'moderate').toLowerCase();
  }

  setElemText('res-conf-tier', res.confidence_tier || (res.confidence >= 80 ? 'High confidence' : res.confidence >= 60 ? 'Moderate confidence' : 'Low confidence'));
  setElemText('res-conf-ascii', res.confidence_ascii || '');

  const confBarElem = document.getElementById('res-bar-confidence');
  if (confBarElem) {
    confBarElem.style.width = res.confidence + '%';
    confBarElem.style.backgroundColor = res.confidence >= 80 ? '#2e7d32' : res.confidence >= 60 ? '#ef6c00' : '#d32f2f';
  }

  const warningBox = document.getElementById('res-uncertainty-warning');
  if (warningBox) {
    warningBox.style.display = (res.confidence_warning || res.confidence < 60.0) ? 'block' : 'none';
  }

  const heatmapImg = document.getElementById('gradcam-heatmap-img');
  if (heatmapImg && res.gradcam_heatmap_url) {
    heatmapImg.src = res.gradcam_heatmap_url;
    heatmapImg.style.display = 'block';
  }
  setElemText('gradcam-explanation-text', res.gradcam_explanation || '');

  const actionsUl = document.getElementById('res-actions-ul');
  if (actionsUl && res.actions) {
    actionsUl.innerHTML = res.actions.map(act => `<li>${act}</li>`).join('');
  }

  const prevUl = document.getElementById('res-prevention-ul');
  if (prevUl && res.prevention) {
    prevUl.innerHTML = res.prevention.map(pr => `<li>${pr}</li>`).join('');
  }
}

async function loadEvaluationMetrics() {
  try {
    const res = await fetch('/api/evaluate');
    const data = await res.json();
    setElemText('eval-train-acc', data.train_accuracy + '%');
    setElemText('eval-val-acc', data.val_accuracy + '%');
    setElemText('eval-test-acc', data.test_accuracy + '%');
    setElemText('eval-precision', data.precision + '%');
    setElemText('eval-recall', data.recall + '%');
    setElemText('eval-f1', data.f1_score + '%');
  } catch (err) {
    console.error('Error fetching evaluation metrics:', err);
  }
}

// --- Farmer Assistant Panel (PDF RAG) ---
function askQuickQuestion(text) {
  const input = document.getElementById('assistant-input');
  if (input) input.value = text;
  sendFarmerQuery();
}

function handleAssistantKeyPress(event) {
  if (event.key === 'Enter') {
    sendFarmerQuery();
  }
}

async function sendFarmerQuery() {
  const input = document.getElementById('assistant-input');
  const query = input ? input.value.trim() : '';
  if (!query) return;

  const chatBox = document.getElementById('assistant-chat-box');
  if (chatBox) {
    chatBox.innerHTML += `
      <div class="chat-message user">
        <div class="msg-bubble">${query}</div>
      </div>
    `;
    if (input) input.value = '';
    chatBox.scrollTop = chatBox.scrollHeight;
  }

  try {
    const res = await fetch('/api/farmer_assistant', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: query, language: currentLang })
    });
    const data = await res.json();

    let citationHtml = '';
    if (data.sources && data.sources.length > 0) {
      citationHtml = `<div style="margin-top: 8px; font-size: 0.78rem; color: var(--secondary-color); font-weight: 600; border-top: 1px dashed var(--border-color); padding-top: 6px;">
        <i class="fa-solid fa-book-bookmark"></i> Sources: ${data.sources.join(', ')}
      </div>`;
    }

    if (chatBox) {
      chatBox.innerHTML += `
        <div class="chat-message assistant">
          <div class="msg-category"><i class="fa-solid fa-shield-cat"></i> ${data.category}</div>
          <div class="msg-bubble">
            ${data.answer}
            ${citationHtml}
          </div>
        </div>
      `;
      chatBox.scrollTop = chatBox.scrollHeight;
    }
  } catch (err) {
    console.error('Assistant error:', err);
  }
}

// --- Dynamic User Task CRUD & Queue Management ---
async function loadUserTaskQueue() {
  try {
    const res = await fetch('/api/scheduler/tasks');
    const data = await res.json();
    renderUserTaskQueue(data.tasks);
  } catch (err) {
    console.error('Error loading task queue:', err);
  }
}

function renderUserTaskQueue(tasks) {
  const tbody = document.getElementById('task-queue-tbody');
  if (!tbody) return;
  if (!tasks || tasks.length === 0) {
    tbody.innerHTML = '<tr><td colspan="8" style="text-align: center; color: var(--text-muted);">No active tasks in user queue. Add a custom task above.</td></tr>';
    return;
  }

  tbody.innerHTML = tasks.map(t => `
    <tr>
      <td><strong>${t.task_id}</strong></td>
      <td>${t.task_type}</td>
      <td>${t.crop || 'Tomato'}</td>
      <td>${t.disease_risk !== undefined ? t.disease_risk : '0.5'}</td>
      <td><span class="severity-badge severity-${(t.severity || 'moderate').toLowerCase()}">${t.severity}</span></td>
      <td>${t.deadline || 15}s</td>
      <td><strong style="color: var(--primary-color);">${t.dynamic_priority ? t.dynamic_priority.toFixed(3) : 'Pending'}</strong></td>
      <td>
        <button class="btn btn-outline" style="padding: 2px 8px; font-size: 0.78rem; color: #d32f2f;" onclick="deleteUserTask('${t.task_id}')">
          <i class="fa-solid fa-trash"></i>
        </button>
      </td>
    </tr>
  `).join('');
}

async function addUserTask() {
  const taskType = document.getElementById('task-type-input') ? document.getElementById('task-type-input').value : 'Leaf_Inference';
  const crop = document.getElementById('task-crop-input') ? document.getElementById('task-crop-input').value : 'Tomato';
  const risk = parseFloat(document.getElementById('task-risk-input') ? document.getElementById('task-risk-input').value : 0.5) || 0.5;
  const severity = document.getElementById('task-severity-input') ? document.getElementById('task-severity-input').value : 'High';
  const cropImp = parseFloat(document.getElementById('task-crop-imp-input') ? document.getElementById('task-crop-imp-input').value : 0.8) || 0.8;
  const deadline = parseFloat(document.getElementById('task-deadline-input') ? document.getElementById('task-deadline-input').value : 15.0) || 15.0;
  const cpu = parseFloat(document.getElementById('task-cpu-input') ? document.getElementById('task-cpu-input').value : 30.0) || 30.0;
  const ram = parseFloat(document.getElementById('task-ram-input') ? document.getElementById('task-ram-input').value : 100.0) || 100.0;

  const newTask = {
    task_id: `TSK-${Math.floor(100 + Math.random() * 900)}`,
    task_type: taskType,
    crop: crop,
    disease_risk: risk,
    severity: severity,
    crop_importance: cropImp,
    deadline: deadline,
    cpu_req: cpu,
    ram_req: ram,
    arrival_time: 0.0,
    processing_time: 1.5,
    net_req: 10.0,
    priority: 1
  };

  try {
    const res = await fetch('/api/scheduler/tasks', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(newTask)
    });
    const data = await res.json();
    if (data.success) {
      loadUserTaskQueue();
      triggerSchedulerRun();
    }
  } catch (err) {
    alert('Error adding user task: ' + err.message);
  }
}

async function deleteUserTask(taskId) {
  try {
    const res = await fetch(`/api/scheduler/tasks?task_id=${encodeURIComponent(taskId)}`, { method: 'DELETE' });
    const data = await res.json();
    if (data.success) {
      loadUserTaskQueue();
      triggerSchedulerRun();
    }
  } catch (err) {
    console.error('Error deleting task:', err);
  }
}

async function clearUserTaskQueue() {
  try {
    const res = await fetch('/api/scheduler/tasks', { method: 'DELETE' });
    const data = await res.json();
    if (data.success) {
      loadUserTaskQueue();
      triggerSchedulerRun();
    }
  } catch (err) {
    console.error('Error clearing queue:', err);
  }
}

async function runUserQueueScheduler() {
  const algoSelect = document.getElementById('scheduler-algo-select');
  const selectedAlgo = algoSelect ? algoSelect.value : 'Adaptive';

  try {
    const res = await fetch('/api/scheduler/run_user_queue', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ algorithm: selectedAlgo })
    });
    const data = await res.json();
    if (data.success) {
      const taskLog = data.results.task_log;
      renderExecutedTaskLog(taskLog, data.algorithm_executed);
    }
  } catch (err) {
    alert('Error executing scheduler queue: ' + err.message);
  }
}

function renderExecutedTaskLog(taskLog, algoName) {
  const tbody = document.getElementById('task-queue-tbody');
  if (!tbody) return;
  if (!taskLog || taskLog.length === 0) {
    tbody.innerHTML = '<tr><td colspan="8" style="text-align: center; color: var(--text-muted);">No tasks executed.</td></tr>';
    return;
  }

  tbody.innerHTML = taskLog.map((t, idx) => `
    <tr style="${idx === 0 ? 'background: #e8f5e9;' : ''}">
      <td><strong>${t.task_id}</strong></td>
      <td>${t.type}</td>
      <td>${t.crop}</td>
      <td>${t.disease_risk !== undefined ? t.disease_risk : '0.5'}</td>
      <td><span class="severity-badge severity-${(t.severity || 'moderate').toLowerCase()}">${t.severity}</span></td>
      <td>${t.deadline_sec || 15}s</td>
      <td><strong style="color: var(--primary-color);">${t.dynamic_priority ? t.dynamic_priority.toFixed(3) : idx + 1}</strong></td>
      <td>
        <span class="status-badge ${idx === 0 ? 'status-online' : 'status-offline'}">
          ${idx === 0 ? 'EXECUTED #1' : 'SEQUENCE #' + (idx + 1)}
        </span>
      </td>
    </tr>
  `).join('');
}

// --- Resource Scheduler & Concurrency ---
async function triggerSchedulerRun() {
  try {
    const res = await fetch('/api/compare_schedulers');
    const data = await res.json();
    renderDeadlockState();
  } catch (err) {
    console.error('Scheduler run error:', err);
  }
}

// --- Deadlock Arbitration ---
async function runDeadlockCheck(scenario = 'safe') {
  try {
    const res = await fetch(`/api/deadlock_check?scenario=${scenario}`);
    const data = await res.json();

    const box = document.getElementById('deadlock-status-box');
    if (box) {
      box.innerText = `System State: ${data.status} — ${data.arbitration_result.message}`;
      box.style.background = data.is_safe ? '#e8f5e9' : '#ffebee';
      box.style.color = data.is_safe ? '#2e7d32' : '#c62828';
    }

    const matrixElem = document.getElementById('rag-matrix-display');
    if (matrixElem) {
      matrixElem.innerHTML = `Available Vector [CPU, RAM, ML, Net]: [${data.available.join(', ')}]\n\nAllocation Matrix:\n${data.allocation_matrix.map(a => `  ${a.process}: [${a.alloc.join(', ')}]`).join('\n')}`;
    }
  } catch (err) {
    console.error('Deadlock check error:', err);
  }
}

function renderDeadlockState() {
  runDeadlockCheck('safe');
}

// --- Scheduler Algorithm Comparison Benchmark ---
async function runAlgorithmComparison() {
  const tbody = document.getElementById('comparison-tbody');
  if (tbody) {
    tbody.innerHTML = '<tr><td colspan="6" style="text-align: center;"><i class="fa-solid fa-spinner fa-spin"></i> Running 25-task workload across 5 algorithms...</td></tr>';
  }

  try {
    const res = await fetch('/api/compare_schedulers');
    const data = await res.json();
    const comp = data.comparison;

    const algos = ['FCFS', 'RoundRobin', 'Priority', 'EDF', 'Adaptive'];
    if (tbody) {
      tbody.innerHTML = algos.map(name => {
        const item = comp[name];
        const isAdaptive = (name === 'Adaptive');
        return `
          <tr style="${isAdaptive ? 'background-color: #e8f5e9; font-weight: 700;' : ''}">
            <td><strong>${name === 'Adaptive' ? 'Proposed Adaptive (Novel)' : name}</strong></td>
            <td>${item.avg_waiting_time_sec} s</td>
            <td>${item.avg_response_time_sec} s</td>
            <td>${item.throughput_tasks_per_sec} tasks/s</td>
            <td>${item.deadline_miss_rate_pct} % (${item.deadline_misses})</td>
            <td><strong style="color: #2e7d32;">${item.critical_disease_delay_sec} s</strong></td>
          </tr>
        `;
      }).join('');
    }
  } catch (err) {
    alert('Comparison benchmark error: ' + err.message);
  }
}

// --- Connectivity Store-and-Forward Toggle ---
async function toggleConnectivity() {
  isOnline = !isOnline;
  const badge = document.getElementById('connection-status-badge');
  const label = document.getElementById('connectivity-label');

  if (badge) badge.className = `status-badge ${isOnline ? 'status-online' : 'status-offline'}`;
  if (label) label.innerText = isOnline ? 'Online (Edge-to-Cloud Sync Active)' : 'Offline (Store-and-Forward)';

  try {
    const res = await fetch('/api/toggle_sync', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ online: isOnline, bandwidth_kbps: isOnline ? 180.0 : 0.0 })
    });
    const data = await res.json();
    if (isOnline && data.success) {
      alert(`Cloud Synchronization complete: ${data.synced_count} records uploaded to Cloud S3.`);
    }
  } catch (err) {
    console.error('Sync toggle error:', err);
  }
}

// --- Storage Dashboard Auto-Archive ---
async function triggerAutoArchive() {
  try {
    const res = await fetch('/api/storage_stats');
    const stats = await res.json();
    setElemText('file-count-pending', stats.pending.file_count);
    setElemText('file-count-processing', stats.processing.file_count);
    setElemText('file-count-completed', stats.completed.file_count);
    setElemText('file-count-critical', stats.critical.file_count);
    setElemText('file-count-archive', stats.archive.file_count);
    alert(`File storage audit completed. Total storage used: ${stats.summary.total_size_mb} MB.`);
  } catch (err) {
    console.error('Storage stats error:', err);
  }
}

function toggleAccordion(header) {
  if (header && header.parentElement) {
    header.parentElement.classList.toggle('open');
  }
}
