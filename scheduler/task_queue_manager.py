"""
Task Queue Manager: Connects Live Application Events to the Scheduler Queue in SQLite.
Whenever an image is uploaded, a farmer asks a question, weather is updated, or sync is requested,
an actual AgriculturalTask is registered and enqueued.
"""

import time
import uuid
from typing import Dict, Any, List
from database.db import get_connection

TASK_PROFILES = {
    "Leaf_Inference": {"burst": 2.5, "cpu": 45.0, "ram": 180.0, "net": 10.0, "deadline": 15.0, "base_priority": 1},
    "Farmer_RAG":     {"burst": 1.2, "cpu": 25.0, "ram": 95.0,  "net": 5.0,  "deadline": 20.0, "base_priority": 2},
    "Cloud_Sync":     {"burst": 3.0, "cpu": 15.0, "ram": 35.0,  "net": 100.0,"deadline": 60.0, "base_priority": 4},
    "Weather_Check":  {"burst": 0.5, "cpu": 10.0, "ram": 20.0,  "net": 15.0, "deadline": 30.0, "base_priority": 3},
    "Sensor_Process": {"burst": 0.4, "cpu": 8.0,  "ram": 15.0,  "net": 5.0,  "deadline": 45.0, "base_priority": 3}
}

def enqueue_application_task(task_type: str, crop: str = "General", severity: str = "Moderate", disease_risk: float = 0.5, device_id: str = "Camera-01") -> str:
    """Enqueues an actual application event into SQLite task_queue."""
    conn = get_connection()
    cursor = conn.cursor()
    
    task_id = f"APP-{task_type[:4].upper()}-{str(uuid.uuid4())[:6]}"
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    
    profile = TASK_PROFILES.get(task_type, TASK_PROFILES["Sensor_Process"])
    
    crop_importance = 0.9 if crop in ["Tomato", "Potato", "Rice"] else 0.75
    urgency = 0.8 if severity in ["Critical", "High"] else 0.4
    
    cursor.execute("""
    INSERT OR REPLACE INTO task_queue (
        task_id, task_type, device_id, arrival_time, processing_time,
        cpu_req, ram_req, net_req, priority, disease_risk, crop_importance,
        urgency, deadline, status, executed_by_algorithm
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, (
        task_id, task_type, device_id, now_str, profile["burst"],
        profile["cpu"], profile["ram"], profile["net"], profile["base_priority"],
        disease_risk, crop_importance, urgency, profile["deadline"], "QUEUED", "PENDING"
    ))
    
    conn.commit()
    conn.close()
    return task_id

def get_live_task_queue(limit: int = 15) -> List[Dict[str, Any]]:
    """Retrieve current queued tasks from SQLite."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM task_queue ORDER BY rowid DESC LIMIT ?;", (limit,))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows
