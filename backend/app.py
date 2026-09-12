"""
Flask Web Application Server for Agricultural Edge Intelligence OS.
Provides REST APIs for Disease Detection, OS Resource Schedulers, Memory Management,
Deadlock Arbitration, Storage Management, and Farmer Assistance.
"""

import os
import sys
import json
import random
from flask import Flask, request, jsonify, send_from_directory

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from database.db import init_db, save_leaf_scan, get_all_scans, get_dashboard_summary, save_farmer_query
from ml.inference import run_disease_inference
from ml.knowledge_base import KNOWLEDGE_BASE
from scheduler.adaptive_scheduler import compare_all_schedulers, run_scheduler_simulation, generate_sample_workload
from memory.memory_manager import memory_manager
from storage.file_manager import store_new_image, transition_file_status, get_storage_stats, archive_old_files
from deadlock.deadlock_arbitrator import simulate_deadlock_scenario
from backend.sync_manager import sync_manager
from backend.config import DEFAULT_WEATHER, TRANSLATIONS_DIR, HOST, PORT, DEBUG

app = Flask(__name__, static_folder=os.path.join(PROJECT_ROOT, "frontend"))

# Initialize SQLite Database on startup
init_db()


# --- Frontend Routes ---

@app.route('/')
def index():
    """Serve the main Agricultural Intelligence Dashboard."""
    return send_from_directory(os.path.join(PROJECT_ROOT, 'frontend'), 'index.html')

@app.route('/css/<path:filename>')
def serve_css(filename):
    return send_from_directory(os.path.join(PROJECT_ROOT, 'frontend', 'css'), filename)

@app.route('/js/<path:filename>')
def serve_js(filename):
    return send_from_directory(os.path.join(PROJECT_ROOT, 'frontend', 'js'), filename)

@app.route('/data/<path:filename>')
def serve_uploaded_data(filename):
    return send_from_directory(os.path.join(PROJECT_ROOT, 'data'), filename)


# --- REST API Endpoints ---

@app.route('/api/translate/<lang>', methods=['GET'])
def get_translation(lang):
    """Retrieve translation dictionary for specified language code."""
    file_path = os.path.join(TRANSLATIONS_DIR, f"{lang}.json")
    if not os.path.exists(file_path):
        file_path = os.path.join(TRANSLATIONS_DIR, "en.json")
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return jsonify(data)


@app.route('/api/analyze_disease', methods=['POST'])
def analyze_disease():
    """
    Core Feature 1: Leaf Image Upload & Real EfficientNet-B0 Disease Detection.
    Stores image in /data/pending, allocates memory, runs inference, and transitions file state.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "Selected file is empty"}), 400

    model_type = request.form.get('model', 'efficientnet_b0')
    device_id = request.form.get('device_id', 'Edge-Camera-01')

    # Step 1: File Storage - Store in /data/pending
    file_bytes = image_file.read()
    img_path, clean_filename, meta = store_new_image(file_bytes, image_file.filename, device_id=device_id)

    # Step 2: Memory Management - Allocate RAM for Inference
    model_name = "ResNet-50" if model_type == "resnet50" else "EfficientNet-B0"
    mem_result = memory_manager.allocate(model_name)

    # Step 3: Transition File State to /data/processing
    transition_file_status(clean_filename, "processing")

    # Step 4: ML Inference Engine Execution
    image_file.seek(0)
    diagnosis = run_disease_inference(image_file, model_type=model_type)

    # Step 5: Transition File State to /data/completed or /data/critical
    final_state = "critical" if diagnosis["severity"] in ["Critical", "High"] else "completed"
    transition_file_status(clean_filename, final_state, diagnosis_result=diagnosis)

    # Step 6: Save Record in SQLite Database
    scan_id = save_leaf_scan(
        crop=diagnosis["crop"],
        disease=diagnosis["disease"],
        confidence=diagnosis["confidence"],
        severity=diagnosis["severity"],
        is_healthy=diagnosis["is_healthy"],
        status=final_state,
        image_filename=clean_filename,
        file_path=f"/data/{final_state}/{clean_filename}",
        temp=DEFAULT_WEATHER["temperature"],
        humidity=DEFAULT_WEATHER["humidity"],
        rain_prob=DEFAULT_WEATHER["rain_probability"],
        device_id=device_id
    )

    diagnosis["scan_id"] = scan_id
    diagnosis["image_url"] = f"/data/{final_state}/{clean_filename}"
    diagnosis["memory_status"] = mem_result
    diagnosis["storage_state"] = final_state

    return jsonify(diagnosis)


@app.route('/api/dashboard_summary', methods=['GET'])
def dashboard_summary():
    """Retrieve Overview Metrics, Environmental Climate Context, and System Snapshot."""
    summary = get_dashboard_summary()
    scans = get_all_scans()
    storage = get_storage_stats()
    mem_state = memory_manager.get_state()

    # Weather Risk Calculation
    humidity = DEFAULT_WEATHER["humidity"]
    rain_prob = DEFAULT_WEATHER["rain_probability"]
    fungal_risk = "HIGH" if (humidity > 80 and rain_prob > 50) else "MODERATE"

    return jsonify({
        "summary": summary,
        "recent_scans": scans[:10],
        "weather": {
            **DEFAULT_WEATHER,
            "fungal_disease_risk": fungal_risk
        },
        "system_resources": {
            "cpu_utilization_pct": round(random.uniform(22.0, 48.0), 1),
            "memory": mem_state,
            "storage": storage["summary"],
            "network_kbps": sync_manager.bandwidth_kbps,
            "is_online": sync_manager.is_online,
            "battery_pct": 88.0
        }
    })


@app.route('/api/compare_schedulers', methods=['GET'])
def compare_schedulers_api():
    """
    Run 5-Algorithm Comparison Benchmark (FCFS, RR, Priority, EDF, Proposed Adaptive).
    Returns comparative metrics & charts data.
    """
    results = compare_all_schedulers(weather_context=DEFAULT_WEATHER)
    return jsonify({
        "success": True,
        "workload_task_count": 25,
        "comparison": results
    })


@app.route('/api/deadlock_check', methods=['GET', 'POST'])
def deadlock_check_api():
    """Execute Banker's Algorithm Deadlock & Resource Arbitration Simulation."""
    scenario = request.args.get('scenario', 'safe')
    result = simulate_deadlock_scenario(scenario_type=scenario)
    return jsonify(result)


@app.route('/api/farmer_assistant', methods=['POST'])
def farmer_assistant_api():
    """Controlled Non-Hallucinated Farmer Assistance Panel."""
    data = request.json or {}
    query = data.get('query', '').strip()
    lang = data.get('language', 'en')

    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    q_lower = query.lower()
    
    # Controlled Agricultural Knowledge Answers
    if "water" in q_lower or "irrigation" in q_lower:
        category = "Irrigation Management"
        answer = "Irrigate early in the morning near the root zone. Avoid overhead sprinkling as leaf wetness increases fungal spore germination (Early/Late Blight)."
    elif "fertilizer" in q_lower or "nutrient" in q_lower:
        category = "Nutrient Management"
        answer = "Apply balanced N-P-K (10-26-26 or 19-19-19). Avoid excess nitrogen during humid weather as it encourages soft succulent leaves vulnerable to bacterial blight."
    elif "prevent" in q_lower or "protect" in q_lower or "spread" in q_lower:
        category = "Disease Prevention"
        answer = "Prune lower infected leaves, apply organic mulch to limit soil splash, ensure 45-60cm plant spacing for ventilation, and spray preventative copper fungicide."
    elif "rain" in q_lower or "weather" in q_lower:
        category = "Weather Impact"
        answer = "High humidity (>80%) combined with rain creates high fungal risk. Apply protective contact fungicide (Mancozeb) before expected rain spells."
    elif "animal" in q_lower or "safe" in q_lower or "cattle" in q_lower:
        category = "Safety & Livestock"
        answer = "Do not allow livestock to feed on foliage recently sprayed with chemical fungicides. Observe a minimum 7-14 day pre-harvest/grazing safety interval."
    else:
        category = "General Crop Care"
        answer = "Sanitize farm tools with alcohol solution, remove diseased plant residue immediately, and practice 2-3 year crop rotation with non-host crops."

    save_farmer_query(query, category, answer, language=lang)

    return jsonify({
        "query": query,
        "category": category,
        "answer": answer,
        "timestamp": DEFAULT_WEATHER.get("timestamp", "Just now")
    })


@app.route('/api/toggle_sync', methods=['POST'])
def toggle_sync_api():
    """Toggle online/offline mode and trigger store-and-forward cloud sync."""
    data = request.json or {}
    online = data.get('online', True)
    bw = data.get('bandwidth_kbps', 150.0)

    sync_manager.set_connectivity(online, bw)
    res = sync_manager.trigger_sync()
    return jsonify(res)


@app.route('/api/storage_stats', methods=['GET'])
def storage_stats_api():
    """Retrieve File Hierarchy Storage Status."""
    stats = get_storage_stats()
    return jsonify(stats)


if __name__ == '__main__':
    print(f"Starting AgriEdge Intelligence Server on http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=DEBUG)
