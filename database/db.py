"""
SQLite Database Access & Schema Initialization.
Provides thread-safe storage for leaf scans, task queues, farmer queries, and resource metrics.
"""

import sqlite3
import os
import json
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "agri_edge_os.db")

def get_connection():
    """Get a thread-safe connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database tables and default sample records."""
    conn = get_connection()
    cursor = conn.cursor()

    # Leaf Scans Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS leaf_scans (
        scan_id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        crop TEXT NOT NULL,
        disease TEXT NOT NULL,
        confidence REAL NOT NULL,
        severity TEXT NOT NULL,
        is_healthy INTEGER NOT NULL,
        status TEXT NOT NULL,
        image_filename TEXT NOT NULL,
        file_path TEXT NOT NULL,
        temperature REAL,
        humidity REAL,
        rain_probability REAL,
        device_id TEXT DEFAULT 'Edge-Node-01'
    );
    """)

    # Task Queue Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS task_queue (
        task_id TEXT PRIMARY KEY,
        task_type TEXT NOT NULL,
        device_id TEXT NOT NULL,
        arrival_time TEXT NOT NULL,
        processing_time REAL NOT NULL,
        cpu_req REAL NOT NULL,
        ram_req REAL NOT NULL,
        net_req REAL NOT NULL,
        priority REAL NOT NULL,
        disease_risk REAL NOT NULL,
        crop_importance REAL NOT NULL,
        urgency REAL NOT NULL,
        deadline REAL NOT NULL,
        status TEXT NOT NULL,
        executed_by_algorithm TEXT
    );
    """)

    # Farmer Queries Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS farmer_queries (
        query_id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        question TEXT NOT NULL,
        category TEXT NOT NULL,
        answer TEXT NOT NULL,
        language TEXT DEFAULT 'en'
    );
    """)

    # Sync Log Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sync_log (
        sync_id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        items_count INTEGER NOT NULL,
        status TEXT NOT NULL,
        bandwidth_kbps REAL NOT NULL
    );
    """)

    conn.commit()
    
    # Populate initial sample scan history if empty
    cursor.execute("SELECT COUNT(*) FROM leaf_scans;")
    if cursor.fetchone()[0] == 0:
        sample_scans = [
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Tomato", "Early Blight", 94.2, "Moderate", 0, "completed", "sample_leaf_1.jpg", "/data/completed/sample_leaf_1.jpg", 28.5, 78.0, 15.0, "Camera-01"),
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Potato", "Late Blight", 96.8, "Critical", 0, "critical", "sample_leaf_2.jpg", "/data/critical/sample_leaf_2.jpg", 29.1, 84.5, 45.0, "Drone-01"),
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Maize", "Healthy Leaf", 98.4, "None", 1, "completed", "sample_leaf_3.jpg", "/data/completed/sample_leaf_3.jpg", 26.0, 65.0, 5.0, "Camera-02"),
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Rice", "Bacterial Leaf Blight", 91.5, "High", 0, "completed", "sample_leaf_4.jpg", "/data/completed/sample_leaf_4.jpg", 31.0, 88.0, 60.0, "Camera-03"),
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Cotton", "Healthy Leaf", 97.1, "None", 1, "completed", "sample_leaf_5.jpg", "/data/completed/sample_leaf_5.jpg", 30.2, 58.0, 0.0, "Camera-01")
        ]
        cursor.executemany("""
        INSERT INTO leaf_scans (timestamp, crop, disease, confidence, severity, is_healthy, status, image_filename, file_path, temperature, humidity, rain_probability, device_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, sample_scans)
        conn.commit()

    conn.close()

def save_leaf_scan(crop, disease, confidence, severity, is_healthy, status, image_filename, file_path, temp=28.0, humidity=75.0, rain_prob=20.0, device_id="Edge-Node-01"):
    """Insert a new leaf scan record into the database."""
    conn = get_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute("""
    INSERT INTO leaf_scans (timestamp, crop, disease, confidence, severity, is_healthy, status, image_filename, file_path, temperature, humidity, rain_probability, device_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, (now_str, crop, disease, confidence, severity, 1 if is_healthy else 0, status, image_filename, file_path, temp, humidity, rain_prob, device_id))
    
    scan_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return scan_id

def get_all_scans():
    """Retrieve all leaf scan records sorted by timestamp descending."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM leaf_scans ORDER BY scan_id DESC LIMIT 100;")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def get_dashboard_summary():
    """Calculate summary statistics for dashboard overview metrics."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM leaf_scans;")
    total = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM leaf_scans WHERE is_healthy = 1;")
    healthy = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM leaf_scans WHERE is_healthy = 0;")
    diseased = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM leaf_scans WHERE severity = 'Critical';")
    critical = cursor.fetchone()[0]
    
    cursor.execute("SELECT AVG(confidence) FROM leaf_scans;")
    avg_conf = cursor.fetchone()[0] or 0.0
    
    conn.close()
    
    return {
        "total_analyzed": total,
        "healthy_count": healthy,
        "diseased_count": diseased,
        "critical_count": critical,
        "avg_confidence": round(avg_conf, 1)
    }

def save_farmer_query(question, category, answer, language="en"):
    """Store farmer assistant question and response."""
    conn = get_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
    INSERT INTO farmer_queries (timestamp, question, category, answer, language)
    VALUES (?, ?, ?, ?, ?);
    """, (now_str, question, category, answer, language))
    conn.commit()
    conn.close()

# --- Task Queue CRUD Operations ---
def get_user_tasks():
    """Retrieve all current tasks from SQLite task queue."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM task_queue ORDER BY arrival_time ASC;")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def save_user_task(task_dict):
    """Insert or update a user task in SQLite task queue."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT OR REPLACE INTO task_queue 
    (task_id, task_type, device_id, arrival_time, processing_time, cpu_req, ram_req, net_req, priority, disease_risk, crop_importance, urgency, deadline, status)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, (
        task_dict["task_id"],
        task_dict["task_type"],
        task_dict.get("device_id", "Camera-01"),
        str(task_dict.get("arrival_time", 0.0)),
        float(task_dict.get("processing_time", 1.5)),
        float(task_dict.get("cpu_req", 25.0)),
        float(task_dict.get("ram_req", 60.0)),
        float(task_dict.get("net_req", 10.0)),
        float(task_dict.get("priority", 1.0)),
        float(task_dict.get("disease_risk", 0.5)),
        float(task_dict.get("crop_importance", 0.8)),
        float(task_dict.get("urgency", 0.5)),
        float(task_dict.get("deadline", 15.0)),
        task_dict.get("status", "Pending")
    ))
    conn.commit()
    conn.close()

def delete_user_task(task_id):
    """Delete a task by ID from SQLite task queue."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM task_queue WHERE task_id = ?;", (task_id,))
    conn.commit()
    conn.close()

def clear_user_tasks():
    """Remove all tasks from SQLite task queue."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM task_queue;")
    conn.commit()
    conn.close()

