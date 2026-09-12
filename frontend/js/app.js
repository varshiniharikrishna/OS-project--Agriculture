/* ==========================================================================
   AgriEdge Intelligence OS — Client Application Logic
   Handles Multilingual UI, Leaf Diagnostics, Schedulers, and OS Visualizations
   ========================================================================== */

let currentLang = 'en';
let i18nDict = {};
let selectedFile = null;
let isOnline = false;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
  loadDashboardData();
  changeLanguage('en');
  triggerSchedulerRun();
});

// --- Section Navigation ---
function showSection(sectionId) {
  document.querySelectorAll('.page-section').forEach(sec => sec.classList.remove('active'));
  document.querySelectorAll('.sidebar-menu li').forEach(item => item.classList.remove('active'));

  const targetSec = document.getElementById(`section-${sectionId}`);
  if (targetSec) {
    targetSec.classList.add('active');
  }

  // Highlight menu item
  const activeLink = document.querySelector(`.sidebar-menu a[href="#section-${sectionId}"]`);
  if (activeLink && activeLink.parentElement) {
    activeLink.parentElement.classList.add('active');
  }

  // Update header title
  const titleMap = {
    'dashboard': 'Dashboard Overview',
    'detection': 'Crop Disease Detection',
    'assistant': 'Farmer Assistance',
    'scheduler': 'Resource Scheduler',
    'resources': 'Resource & Weather Monitor',
    'comparison': 'Scheduler Comparison',
    'files': 'File Storage Management',
    'technical': 'Technical Details'
  };
  document.getElementById('current-section-title').innerText = titleMap[sectionId] || 'Dashboard';
}

// --- Multilingual Translation ---
async function changeLanguage(langCode) {
  currentLang = langCode;
  try {
    const res = await fetch(`/api/translate/${langCode}`);
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
      elem.innerText = i18nDict[key];
    }
  });
}

// --- Data Fetching & Dashboard Load ---
async function loadDashboardData() {
  try {
    const res = await fetch('/api/dashboard_summary');
    const data = await res.json();

    // Summary Cards
    document.getElementById('stat-total').innerText = data.summary.total_analyzed;
    document.getElementById('stat-healthy').innerText = data.summary.healthy_count;
    document.getElementById('stat-diseased').innerText = data.summary.diseased_count;
    document.getElementById('stat-critical').innerText = data.summary.critical_count;
    document.getElementById('stat-confidence').innerText = `${data.summary.avg_confidence}%`;

    // Climate Context
    document.getElementById('dash-weather-temp').innerText = `${data.weather.temperature} °C`;
    document.getElementById('dash-weather-humidity').innerText = `${data.weather.humidity} %`;
    document.getElementById('dash-weather-rain').innerText = `${data.weather.rain_probability} %`;
    const riskElem = document.getElementById('dash-weather-risk');
    riskElem.innerText = `${data.weather.fungal_disease_risk} RISK`;
    riskElem.style.color = data.weather.fungal_disease_risk === 'HIGH' ? '#d32f2f' : '#2e7d32';

    // Resource Gauges
    document.getElementById('res-val-cpu').innerText = `${data.system_resources.cpu_utilization_pct} %`;
    document.getElementById('res-bar-cpu').style.width = `${data.system_resources.cpu_utilization_pct}%`;

    const mem = data.system_resources.memory;
    document.getElementById('res-val-ram').innerText = `${mem.used_ram_mb} MB / ${mem.total_ram_mb} MB`;
    document.getElementById('res-bar-ram').style.width = `${mem.utilization_pct}%`;

    const stor = data.system_resources.storage;
    document.getElementById('res-val-storage').innerText = `${stor.total_size_mb} MB / ${stor.storage_capacity_mb} MB`;
    const storPct = (stor.total_size_mb / stor.storage_capacity_mb) * 100;
    document.getElementById('res-bar-storage').style.width = `${Math.max(2, storPct)}%`;

    renderRecentScans(data.recent_scans);
    renderMemoryLRU(mem.active_models);
  } catch (err) {
    console.error('Error loading dashboard summary:', err);
  }
}

function renderRecentScans(scans) {
  const tbody = document.getElementById('recent-scans-tbody');
  if (!scans || scans.length === 0) {
    tbody.innerHTML = '<tr><td colspan="5" style="text-align: center;">No scans recorded yet.</td></tr>';
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

function renderMemoryLRU(activeModels) {
  const container = document.getElementById('memory-models-container');
  if (!activeModels || activeModels.length === 0) {
    container.innerHTML = '<span style="color: var(--text-muted);">No models currently loaded in RAM.</span>';
    return;
  }

  container.innerHTML = activeModels.map(m => `
    <div style="background: white; border: 1px solid var(--border-color); padding: 8px 14px; border-radius: 6px; font-size: 0.85rem;">
      <i class="fa-solid fa-microchip" style="color: var(--primary-color);"></i> <strong>${m.name}</strong>: ${m.ram_mb} MB
    </div>
  `).join('');
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

  if (typeof fileOrUrl === 'string') {
    previewImg.src = fileOrUrl;
  } else {
    previewImg.src = URL.createObjectURL(fileOrUrl);
  }
  previewBox.style.display = 'block';
}

async function submitLeafImage() {
  if (!selectedFile) {
    alert('Please select or upload a leaf image file first.');
    return;
  }

  const btn = document.getElementById('analyze-btn');
  btn.disabled = true;
  btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Running Inference...';

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
    btn.disabled = false;
    btn.innerHTML = '<i class="fa-solid fa-magnifying-glass"></i> <span data-i18n="btn_analyze">Analyze Crop Disease</span>';
    applyTranslations();
  }
}

function loadSampleImage(type) {
  // Create simulated blob for demonstration sample
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
  document.getElementById('res-crop-name').innerText = res.crop;
  document.getElementById('res-disease-name').innerText = res.disease;
  document.getElementById('res-confidence').innerText = `${res.confidence} %`;
  document.getElementById('res-explanation').innerText = res.explanation;
  document.getElementById('res-model-used').innerText = res.model_name || 'EfficientNet-B0';

  const statusText = document.getElementById('res-status-text');
  statusText.innerText = res.is_healthy ? 'Healthy Leaf' : 'Diseased Leaf';
  statusText.style.color = res.is_healthy ? '#2e7d32' : '#d32f2f';

  const badge = document.getElementById('res-severity-badge');
  badge.innerText = `${res.severity} Severity`;
  badge.className = `severity-badge severity-${(res.severity || 'moderate').toLowerCase()}`;

  // Actions UL
  const actionsUl = document.getElementById('res-actions-ul');
  actionsUl.innerHTML = res.actions.map(act => `<li>${act}</li>`).join('');

  // Prevention UL
  const prevUl = document.getElementById('res-prevention-ul');
  prevUl.innerHTML = res.prevention.map(pr => `<li>${pr}</li>`).join('');
}

// --- Farmer Assistant Panel ---
function askQuickQuestion(text) {
  document.getElementById('assistant-input').value = text;
  sendFarmerQuery();
}

function handleAssistantKeyPress(event) {
  if (event.key === 'Enter') {
    sendFarmerQuery();
  }
}

async function sendFarmerQuery() {
  const input = document.getElementById('assistant-input');
  const query = input.value.strip ? input.value.strip() : input.value.trim();
  if (!query) return;

  const chatBox = document.getElementById('assistant-chat-box');
  chatBox.innerHTML += `
    <div class="chat-message user">
      <div class="msg-bubble">${query}</div>
    </div>
  `;

  input.value = '';
  chatBox.scrollTop = chatBox.scrollHeight;

  try {
    const res = await fetch('/api/farmer_assistant', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: query, language: currentLang })
    });
    const data = await res.json();

    chatBox.innerHTML += `
      <div class="chat-message assistant">
        <div class="msg-category"><i class="fa-solid fa-shield-cat"></i> ${data.category}</div>
        <div class="msg-bubble">${data.answer}</div>
      </div>
    `;
    chatBox.scrollTop = chatBox.scrollHeight;
  } catch (err) {
    console.error('Assistant error:', err);
  }
}

// --- Resource Scheduler & Concurrency ---
async function triggerSchedulerRun() {
  try {
    const res = await fetch('/api/compare_schedulers');
    const data = await res.json();
    const adaptiveResult = data.comparison.Adaptive;
    renderTaskQueue(adaptiveResult.task_log);
    renderDeadlockState();
  } catch (err) {
    console.error('Scheduler run error:', err);
  }
}

function renderTaskQueue(taskLog) {
  const tbody = document.getElementById('task-queue-tbody');
  if (!taskLog || taskLog.length === 0) {
    tbody.innerHTML = '<tr><td colspan="6" style="text-align: center;">No pending tasks in queue.</td></tr>';
    return;
  }

  tbody.innerHTML = taskLog.map((t, idx) => `
    <tr>
      <td><strong>${t.task_id}</strong></td>
      <td>${t.type}</td>
      <td>${t.crop}</td>
      <td><strong style="color: var(--primary-color);">${t.dynamic_priority}</strong></td>
      <td><span class="severity-badge severity-${(t.severity || 'moderate').toLowerCase()}">${t.severity}</span></td>
      <td><span class="status-badge ${idx === 0 ? 'status-online' : 'status-offline'}">${idx === 0 ? 'EXECUTING' : 'QUEUED'}</span></td>
    </tr>
  `).join('');
}

// --- Deadlock Arbitration ---
async function runDeadlockCheck(scenario = 'safe') {
  try {
    const res = await fetch(`/api/deadlock_check?scenario=${scenario}`);
    const data = await res.json();

    const box = document.getElementById('deadlock-status-box');
    box.innerText = `System State: ${data.status} — ${data.arbitration_result.message}`;
    box.style.background = data.is_safe ? '#e8f5e9' : '#ffebee';
    box.style.color = data.is_safe ? '#2e7d32' : '#c62828';

    const matrixElem = document.getElementById('rag-matrix-display');
    matrixElem.innerHTML = `
      Available Vector [CPU, RAM, ML, Net]: [${data.available.join(', ')}]\n
      Allocation Matrix:\n${data.allocation_matrix.map(a => `  ${a.process}: [${a.alloc.join(', ')}]`).join('\n')}
    `;
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
  tbody.innerHTML = '<tr><td colspan="6" style="text-align: center;"><i class="fa-solid fa-spinner fa-spin"></i> Running 25-task workload across 5 algorithms...</td></tr>';

  try {
    const res = await fetch('/api/compare_schedulers');
    const data = await res.json();
    const comp = data.comparison;

    const algos = ['FCFS', 'RoundRobin', 'Priority', 'EDF', 'Adaptive'];
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

    renderBenchmarkCharts(comp);
  } catch (err) {
    alert('Comparison benchmark error: ' + err.message);
  }
}

function renderBenchmarkCharts(comp) {
  const algos = ['FCFS', 'RoundRobin', 'Priority', 'EDF', 'Adaptive'];
  const waitingData = algos.map(a => comp[a].avg_waiting_time_sec);
  const criticalData = algos.map(a => comp[a].critical_disease_delay_sec);

  drawCanvasBarChart('chart-waiting', algos, waitingData, 'Avg Waiting Time (s)', '#0288d1');
  drawCanvasBarChart('chart-critical', algos, criticalData, 'Critical Disease Delay (s)', '#2e7d32');
}

function drawCanvasBarChart(canvasId, labels, dataValues, title, color) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const width = canvas.width = canvas.parentElement.clientWidth - 40;
  const height = canvas.height = 200;

  ctx.clearRect(0, 0, width, height);

  const maxVal = Math.max(...dataValues, 10);
  const barWidth = (width - 60) / labels.length;

  labels.forEach((label, idx) => {
    const val = dataValues[idx];
    const barHeight = (val / maxVal) * (height - 60);
    const x = 40 + idx * barWidth + 10;
    const y = height - 30 - barHeight;

    // Draw Bar
    ctx.fillStyle = (label === 'Adaptive') ? '#2e7d32' : color;
    ctx.fillRect(x, y, barWidth - 20, barHeight);

    // Draw Value label
    ctx.fillStyle = '#2c3e50';
    ctx.font = '11px sans-serif';
    ctx.fillText(`${val}s`, x + (barWidth - 20) / 4, y - 5);

    // Draw X Label
    ctx.fillText(label, x, height - 10);
  });
}

// --- Connectivity Store-and-Forward Toggle ---
async function toggleConnectivity() {
  isOnline = !isOnline;
  const badge = document.getElementById('connection-status-badge');
  const label = document.getElementById('connectivity-label');

  badge.className = `status-badge ${isOnline ? 'status-online' : 'status-offline'}`;
  label.innerText = isOnline ? 'Online (Edge-to-Cloud Sync Active)' : 'Offline (Store-and-Forward)';

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
    document.getElementById('file-count-pending').innerText = stats.pending.file_count;
    document.getElementById('file-count-processing').innerText = stats.processing.file_count;
    document.getElementById('file-count-completed').innerText = stats.completed.file_count;
    document.getElementById('file-count-critical').innerText = stats.critical.file_count;
    document.getElementById('file-count-archive').innerText = stats.archive.file_count;
    alert(`File storage audit completed. Total storage used: ${stats.summary.total_size_mb} MB.`);
  } catch (err) {
    console.error('Storage stats error:', err);
  }
}

function toggleAccordion(header) {
  const item = header.parentElement;
  item.classList.toggle('open');
}
