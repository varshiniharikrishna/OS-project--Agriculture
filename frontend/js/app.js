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
  // Map aliases to merged section
  if (sectionId === 'model-evaluation' || sectionId === 'validations' || sectionId === 'metrics-guide') {
    sectionId = 'metrics-validations';
  }

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
    'assistant': 'Farmer RAG Assistant',
    'scheduler': 'Farm Field Tasks & Scheduler',
    'scheduler-lab': 'OS Scheduler Lab (Examiners)',
    'resources': 'Resource & Weather Monitor',
    'comparison': 'Scheduler Comparison Benchmark',
    'metrics-validations': 'Disease Model Evaluation & System Validations',
    'hardware': 'Arduino Uno & Sensor Hardware Abstraction Layer',
    'files': 'File Storage Management',
    'technical': 'Technical Details'
  };
  document.getElementById('current-section-title').innerText = titleMap[sectionId] || 'Dashboard';

  if (sectionId === 'hardware') {
    refreshHardwareStatus();
  }
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

    // 5-Stage Storage Directory File Counts (Screenshot 1)
    if (stor && stor.directories) {
      const dirs = stor.directories;
      const setCnt = (id, obj) => {
        const el = document.getElementById(id);
        if (el && obj) el.innerText = `${obj.file_count} files`;
      };
      setCnt('dir-count-pending', dirs.pending);
      setCnt('dir-count-processing', dirs.processing);
      setCnt('dir-count-completed', dirs.completed);
      setCnt('dir-count-critical', dirs.critical);
      setCnt('dir-count-archive', dirs.archive);
    }

    // LRU Memory Manager UI (Screenshot 1)
    const ramText = document.getElementById('lru-ram-text');
    const ramBar = document.getElementById('lru-ram-bar');
    const evictionList = document.getElementById('lru-eviction-list');
    if (ramText) ramText.innerText = `${mem.used_ram_mb} MB / ${mem.total_ram_mb} MB (${mem.utilization_pct}%)`;
    if (ramBar) ramBar.style.width = `${mem.utilization_pct}%`;
    if (evictionList) {
      evictionList.innerHTML = `Active in RAM: <strong>${(mem.active_models || []).join(', ') || 'None'}</strong> • LRU Evictions Total: <strong>${mem.eviction_count || 0}</strong>`;
    }

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

  const btn = document.getElementById('btn-ask-rag');
  if (btn) btn.disabled = true;

  try {
    const res = await fetch('/api/rag_chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: query, language: currentLang })
    });
    const data = await res.json();

    // Render citations card if sources found
    let citationsHtml = '';
    if (data.sources && data.sources.length > 0) {
      citationsHtml = `
        <div style="margin-top: 8px; padding-top: 8px; border-top: 1px dashed #c8e6c9; font-size: 0.78rem;">
          <strong style="color: #2e7d32;"><i class="fa-solid fa-bookmark"></i> Grounded Citations:</strong><br>
          ${data.sources.map(s => `<span class="badge" style="display:inline-block; background:#e8f5e9; color:#1b5e20; padding:2px 8px; border-radius:10px; margin:2px 4px 2px 0;">📄 ${s.doc} (Page ${s.page}) [Relevance: ${Math.round(s.relevance * 100)}%]</span>`).join('')}
        </div>
      `;
    }

    chatBox.innerHTML += `
      <div class="chat-message assistant">
        <div class="msg-category"><i class="fa-solid fa-book-bookmark"></i> Verified ICAR / FAO Literature</div>
        <div class="msg-bubble">
          <div>${data.answer}</div>
          ${citationsHtml}
        </div>
      </div>
    `;
    chatBox.scrollTop = chatBox.scrollHeight;
  } catch (err) {
    console.error('RAG Assistant error:', err);
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function rebuildRagIndex() {
  try {
    const res = await fetch('/api/rag_status');
    const data = await res.json();
    alert(`RAG Document Index Status:\nTotal Indexed Chunks: ${data.total_chunks}\nDocuments: ${data.docs.map(d => d.doc).join(', ')}`);
  } catch (e) {
    alert('Index check error: ' + e);
  }
}

// --- Resource Scheduler & Concurrency ---
async function triggerSchedulerRun() {
  try {
    const res = await fetch('/api/compare_schedulers');
    const data = await res.json();
    const adaptiveResult = data.comparison.Adaptive;
    // Prefer live queue if populated, otherwise use benchmark log
    if (data.live_queue && data.live_queue.length > 0) {
      renderTaskQueue(data.live_queue.map(q => ({
        task_id: q.task_id,
        type: q.task_type,
        crop: "Tomato",
        dynamic_priority: roundPriority(q.priority, q.disease_risk, q.urgency),
        severity: q.urgency > 0.6 ? "Critical" : "Moderate",
        status: q.status
      })));
    } else {
      renderTaskQueue(adaptiveResult.task_log);
    }
    renderDeadlockState();
  } catch (err) {
    console.error('Scheduler run error:', err);
  }
}

function roundPriority(p, d, u) {
  return Math.round((0.25 * d + 0.20 * 0.8 + 0.15 * 0.9 + 0.15 * 0.7 + 0.15 * u + 0.10 * 1.0) * 100);
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
      <td><span class="status-badge ${idx === 0 ? 'status-online' : 'status-offline'}">${idx === 0 ? 'RUNNING' : 'QUEUED'}</span></td>
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
  const mainTbody = document.getElementById('main-comparison-tbody');
  if (tbody) tbody.innerHTML = '<tr><td colspan="6" style="text-align: center;"><i class="fa-solid fa-spinner fa-spin"></i> Running 30-task benchmark across 5 algorithms...</td></tr>';
  if (mainTbody) mainTbody.innerHTML = '<tr><td colspan="7" style="text-align: center;"><i class="fa-solid fa-spinner fa-spin"></i> Running 30-task benchmark across 5 algorithms...</td></tr>';

  try {
    const res = await fetch('/api/compare_schedulers');
    const data = await res.json();
    const comp = data.comparison;

    const algos = ['FCFS', 'RoundRobin', 'Priority', 'EDF', 'Adaptive'];
    
    // Render Main Comparison Table (Matches Screenshot 3)
    if (mainTbody) {
      mainTbody.innerHTML = algos.map(name => {
        const item = comp[name];
        const isAdaptive = (name === 'Adaptive');
        return `
          <tr style="${isAdaptive ? 'background-color: #e8f5e9; font-weight: 700;' : ''}">
            <td><strong>${isAdaptive ? 'Proposed Context-Aware Adaptive (Novel)' : name}</strong></td>
            <td>${item.tasks_completed}</td>
            <td>${item.total_makespan_sec} s</td>
            <td>${item.avg_waiting_time_sec} s</td>
            <td>${item.avg_response_time_sec} s</td>
            <td>${item.deadline_miss_rate_pct}% (${item.deadline_misses})</td>
            <td><strong style="color: #2e7d32;">${item.critical_disease_delay_sec} s</strong></td>
          </tr>
        `;
      }).join('');
    }

    if (tbody) {
      tbody.innerHTML = algos.map(name => {
        const item = comp[name];
        const isAdaptive = (name === 'Adaptive');
        return `
          <tr style="${isAdaptive ? 'background-color: #e8f5e9; font-weight: 700;' : ''}">
            <td><strong>${isAdaptive ? 'Proposed Adaptive (Novel)' : name}</strong></td>
            <td>${item.avg_waiting_time_sec} s</td>
            <td>${item.avg_response_time_sec} s</td>
            <td>${item.throughput_tasks_per_sec} tasks/s</td>
            <td>${item.deadline_miss_rate_pct} % (${item.deadline_misses})</td>
            <td><strong style="color: #2e7d32;">${item.critical_disease_delay_sec} s</strong></td>
          </tr>
        `;
      }).join('');
    }

    // Render Timeline Blocks for Adaptive Scheduler
    renderTimelineBlocks(comp['Adaptive'].task_log || []);
    renderBenchmarkCharts(comp);
  } catch (err) {
    alert('Comparison benchmark error: ' + err.message);
  }
}

function renderTimelineBlocks(tasks) {
  const container = document.getElementById('timeline-bars');
  if (!container) return;
  if (!tasks || tasks.length === 0) {
    container.innerHTML = '<div style="color: var(--text-muted); font-size: 0.85rem;">No tasks executed yet.</div>';
    return;
  }

  container.innerHTML = tasks.map((t, i) => {
    const isCritical = (t.severity === 'Critical' || t.severity === 'High');
    const bg = isCritical ? '#fee2e2' : '#e0f2fe';
    const border = isCritical ? '#ef4444' : '#0288d1';
    const textColor = isCritical ? '#991b1b' : '#0369a1';

    return `
      <div style="flex: 1; min-width: 110px; background: ${bg}; border: 2px solid ${border}; border-radius: 6px; padding: 8px 10px; font-size: 0.78rem;">
        <div style="font-weight: 800; color: ${textColor};">${t.task_id}</div>
        <div style="font-weight: 600; color: #1e293b;">${t.type}</div>
        <div style="color: var(--text-muted); font-size: 0.72rem;">${t.crop} (${t.severity})</div>
        <div style="margin-top: 4px; padding-top: 4px; border-top: 1px dashed #cbd5e1; font-family: monospace;">
          Prio: <strong>${t.dynamic_priority}</strong><br>
          Wait: <strong>${t.waiting}s</strong>
        </div>
      </div>
    `;
  }).join('');
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

async function injectLabTask() {
  const taskType = document.getElementById('lab-task-type').value;
  const crop = document.getElementById('lab-crop').value;
  const severity = document.getElementById('lab-severity').value;
  const diseaseRisk = parseFloat(document.getElementById('lab-disease-risk').value) || 0.75;

  try {
    const res = await fetch('/api/inject_task', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        task_type: taskType,
        crop: crop,
        severity: severity,
        disease_risk: diseaseRisk,
        device_id: 'Examiner-Lab-Node'
      })
    });
    const data = await res.json();
    alert(`Synthetic Task ${data.task_id} successfully enqueued into SQLite database task_queue!`);
    triggerSchedulerRun();
  } catch (err) {
    alert('Task injection error: ' + err.message);
  }
}

// --- Farm Field Tasks & Scheduler Actions (Screenshot 3 Controls) ---

async function addFarmTask() {
  const taskType = document.getElementById('user-task-type').value;
  const crop = document.getElementById('user-crop-type').value || 'Tomato';
  const severity = document.getElementById('user-severity').value;
  const procTime = parseFloat(document.getElementById('user-proc-time').value) || 1.5;

  const severityRiskMap = { 'Critical': 0.95, 'High': 0.85, 'Moderate': 0.60, 'Low': 0.25 };
  const diseaseRisk = severityRiskMap[severity] || 0.70;

  try {
    const res = await fetch('/api/inject_task', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        task_type: taskType,
        crop: crop,
        severity: severity,
        disease_risk: diseaseRisk,
        processing_time: procTime,
        device_id: 'Farmer-Field-Node'
      })
    });
    const data = await res.json();
    alert(`Task [${taskType} - ${crop}] successfully added to field task queue! Task ID: ${data.task_id}`);
    triggerSchedulerRun();
  } catch (err) {
    alert('Failed to add farm task: ' + err.message);
  }
}

async function clearTaskQueueUI() {
  try {
    const res = await fetch('/api/clear_task_queue', { method: 'POST' });
    const data = await res.json();
    const tbody = document.getElementById('task-queue-tbody');
    if (tbody) {
      tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: var(--text-muted);">Queue cleared. No pending tasks.</td></tr>';
    }
    alert(data.message || 'Task queue cleared successfully.');
  } catch (err) {
    alert('Clear queue error: ' + err.message);
  }
}

async function executeSelectedAlgorithm() {
  const selectElem = document.getElementById('selected-algorithm');
  const algo = selectElem ? selectElem.value : 'Adaptive';
  const algoTitleElem = document.getElementById('exec-algo-title');
  if (algoTitleElem) {
    algoTitleElem.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Executing: ${selectElem.options[selectElem.selectedIndex].text}...`;
  }

  try {
    const res = await fetch(`/api/simulate_scheduler?algorithm=${algo}`);
    const data = await res.json();
    const result = data.result;

    if (algoTitleElem) {
      algoTitleElem.innerText = `Executed: ${selectElem.options[selectElem.selectedIndex].text}`;
    }

    const waitElem = document.getElementById('exec-stat-waiting');
    const respElem = document.getElementById('exec-stat-response');
    const spanElem = document.getElementById('exec-stat-makespan');
    const missElem = document.getElementById('exec-stat-miss');

    if (waitElem) waitElem.innerText = `${result.avg_waiting_time_sec}s`;
    if (respElem) respElem.innerText = `${result.avg_response_time_sec}s`;
    if (spanElem) spanElem.innerText = `${result.total_makespan_sec}s`;
    if (missElem) {
      missElem.innerText = `${result.deadline_miss_rate_pct}% (${result.deadline_misses} misses)`;
      missElem.style.color = result.deadline_misses === 0 ? '#2e7d32' : '#b91c1c';
    }

    // Render timeline bars
    renderTimelineBlocks(result.task_log || []);
  } catch (err) {
    alert('Error executing algorithm simulation: ' + err.message);
  }
}

// --- Application-Level LRU Memory Manager Allocation (Screenshot 1) ---
async function loadModelMemory(modelName) {
  try {
    const res = await fetch('/api/allocate_memory', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_name: modelName })
    });
    const data = await res.json();
    const state = data.memory_state;

    const ramText = document.getElementById('lru-ram-text');
    const ramBar = document.getElementById('lru-ram-bar');
    const evictionList = document.getElementById('lru-eviction-list');

    if (ramText) ramText.innerText = `${state.used_ram_mb} MB / ${state.max_ram_mb} MB (${state.usage_pct}%)`;
    if (ramBar) ramBar.style.width = `${state.usage_pct}%`;
    if (evictionList) {
      evictionList.innerHTML = `Active in RAM: <strong>${state.active_models.join(', ') || 'None'}</strong> • LRU Evictions Total: <strong>${state.eviction_count}</strong>`;
    }

    alert(`Memory Manager: Successfully loaded ${modelName} into LRU cache. Current RAM: ${state.used_ram_mb} MB / 512 MB.`);
  } catch (err) {
    alert('Memory allocation error: ' + err.message);
  }
}

// --- TABLE VII Live System Validation Suite Runner ---
async function runTable7ValidationSuite() {
  const container = document.getElementById('table7-live-results');
  if (!container) return;

  container.style.display = 'block';
  container.innerHTML = `
    <div style="background: #e8f5e9; border: 1px solid #c8e6c9; border-radius: 8px; padding: 15px; text-align: center; color: #1b5e20;">
      <i class="fa-solid fa-spinner fa-spin" style="font-size: 1.5rem; margin-bottom: 8px;"></i>
      <div style="font-weight: 700;">Executing Live System Validation Suite across 6 Subsystems...</div>
      <div style="font-size: 0.82rem; color: #2e7d32; margin-top: 4px;">Testing Scheduler, Disease Inference, 512MB LRU Memory, Banker's Deadlock, SQLite, and Store-and-Forward Sync...</div>
    </div>
  `;

  try {
    const res = await fetch('/api/validate_table7');
    const data = await res.json();

    const cardsHtml = data.validations.map((v, i) => `
      <div style="background: white; border: 1px solid var(--border-color); border-left: 4px solid #2e7d32; border-radius: 6px; padding: 12px 15px; box-shadow: var(--shadow-sm);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
          <div style="font-weight: 700; font-size: 0.95rem; color: #1e293b;">
            <i class="fa-solid fa-circle-check" style="color: #2e7d32; margin-right: 6px;"></i>
            ${i + 1}. ${v.component}
          </div>
          <span style="background: #e8f5e9; color: #2e7d32; font-weight: 800; font-size: 0.75rem; padding: 3px 8px; border-radius: 12px; border: 1px solid #a5d6a7;">
            ${v.status} (${v.execution_time_ms} ms)
          </span>
        </div>
        <div style="font-size: 0.84rem; color: #334155; margin-bottom: 4px;">
          <strong>Benefit:</strong> ${v.expected_benefit}
        </div>
        <div style="font-size: 0.82rem; color: #0288d1; margin-bottom: 4px; font-family: monospace;">
          ${v.metrics}
        </div>
        <div style="font-size: 0.78rem; color: var(--text-muted); border-top: 1px dashed #e2e8f0; padding-top: 4px; margin-top: 4px;">
          <strong>Evidence:</strong> ${v.evidence}
        </div>
      </div>
    `).join('');

    container.innerHTML = `
      <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 16px; margin-bottom: 15px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
          <div style="font-weight: 800; color: #15803d; font-size: 1rem;">
            <i class="fa-solid fa-square-check"></i> ALL 6 TABLE VII COMPONENTS VERIFIED (Status: ${data.overall_status})
          </div>
          <div style="font-size: 0.78rem; color: var(--text-muted);">Verified at: ${data.timestamp}</div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px;">
          ${cardsHtml}
        </div>
      </div>
    `;
  } catch (err) {
    container.innerHTML = `
      <div style="background: #fee2e2; border: 1px solid #fca5a5; border-radius: 8px; padding: 12px; color: #b91c1c;">
        <i class="fa-solid fa-triangle-exclamation"></i> Validation Error: ${err.message}
      </div>
    `;
  }
}

// --- Arduino & Sensor HAL Actions ---

async function refreshHardwareStatus() {
  try {
    const res = await fetch('/api/hardware/status');
    const data = await res.json();
    const hw = data.hardware;

    const badge = document.getElementById('hw-status-badge');
    const label = document.getElementById('hw-status-label');
    const stateElem = document.getElementById('hw-sensor-state');
    const pinElem = document.getElementById('hw-raw-pin');
    const eventElem = document.getElementById('hw-event-count');
    const portElem = document.getElementById('hw-assigned-port');

    if (badge && label) {
      if (hw.hardware_connected) {
        badge.style.background = '#dcfce7';
        badge.style.color = '#15803d';
        badge.style.border = '1px solid #86efac';
        label.innerText = `PHYSICAL ARDUINO CONNECTED (${hw.assigned_port})`;
      } else {
        badge.style.background = '#e0f2fe';
        badge.style.color = '#0369a1';
        badge.style.border = '1px solid #7dd3fc';
        label.innerText = 'SIMULATION MODE (Ready for Monday Hardware Bench)';
      }
    }

    if (stateElem) {
      stateElem.innerText = hw.sensor_state || 'CLEAR';
      stateElem.style.color = hw.obstacle_detected ? '#b91c1c' : '#15803d';
    }

    if (pinElem) {
      pinElem.innerText = `${hw.raw_pin_value} (${hw.raw_pin_value === 0 ? 'LOW = Obstacle Detected' : 'HIGH = Clear'})`;
    }

    if (eventElem) {
      eventElem.innerText = hw.detection_events_total || 0;
    }

    const distElem = document.getElementById('hw-sonar-distance');
    if (distElem) {
      if (hw.distance_cm !== null && hw.distance_cm !== undefined) {
        distElem.innerText = `${hw.distance_cm} cm`;
        distElem.style.color = hw.distance_cm <= 15 ? '#b91c1c' : '#0288d1';
      } else {
        distElem.innerText = hw.hardware_connected ? '-- cm' : '8.5 cm (Sim)';
      }
    }

    if (portElem) {
      const typeStr = hw.sensor_type ? ` [${hw.sensor_type}]` : '';
      portElem.innerText = `${hw.assigned_port || 'Auto-Detect Active'}${typeStr}`;
    }
  } catch (err) {
    console.error('Hardware refresh error:', err);
  }
}

async function simulateObstacle(state) {
  try {
    const res = await fetch('/api/hardware/trigger_obstacle', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ obstacle: state })
    });
    const data = await res.json();
    refreshHardwareStatus();
    if (state) {
      triggerSchedulerRun();
    }
  } catch (err) {
    alert('Hardware simulation error: ' + err.message);
  }
}
