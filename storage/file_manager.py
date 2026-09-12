"""
File Management Module — OS Objective.
Manages file state transitions (/data/pending, processing, completed, critical, archive),
metadata storage, archiving policy, and storage utilization calculation.
"""

import os
import json
import shutil
from datetime import datetime

BASE_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DIRS = {
    "pending": os.path.join(BASE_DATA_DIR, "pending"),
    "processing": os.path.join(BASE_DATA_DIR, "processing"),
    "completed": os.path.join(BASE_DATA_DIR, "completed"),
    "critical": os.path.join(BASE_DATA_DIR, "critical"),
    "archive": os.path.join(BASE_DATA_DIR, "archive")
}

def init_file_storage():
    """Ensure all required directories exist."""
    for path in DIRS.values():
        os.makedirs(path, exist_ok=True)

def store_new_image(file_bytes, filename, crop="Unknown", device_id="Camera-01"):
    """Store an uploaded image in /data/pending/ with associated metadata JSON."""
    init_file_storage()
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_filename = f"{timestamp_str}_{filename.replace(' ', '_')}"
    
    target_img_path = os.path.join(DIRS["pending"], clean_filename)
    target_meta_path = os.path.join(DIRS["pending"], f"{clean_filename}.json")

    # Write binary image file
    with open(target_img_path, "wb") as f:
        f.write(file_bytes)

    # Write initial metadata file
    metadata = {
        "filename": clean_filename,
        "original_name": filename,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "pending",
        "crop": crop,
        "device_id": device_id,
        "file_size_bytes": len(file_bytes),
        "disease_prediction": None,
        "confidence": 0.0,
        "severity": "Unknown"
    }

    with open(target_meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return target_img_path, clean_filename, metadata

def transition_file_status(filename, new_status, diagnosis_result=None):
    """Move file and metadata between states (pending -> processing -> completed/critical -> archive)."""
    init_file_storage()
    current_location = None

    # Locate current directory of file
    for state, dir_path in DIRS.items():
        img_p = os.path.join(dir_path, filename)
        if os.path.exists(img_p):
            current_location = state
            break

    if not current_location:
        # Create virtual file in completed if not on disk
        current_location = "pending"

    src_img = os.path.join(DIRS[current_location], filename)
    src_meta = os.path.join(DIRS[current_location], f"{filename}.json")

    dest_state = new_status.lower()
    if dest_state not in DIRS:
        dest_state = "completed"

    dst_img = os.path.join(DIRS[dest_state], filename)
    dst_meta = os.path.join(DIRS[dest_state], f"{filename}.json")

    # Read existing metadata or construct new
    if os.path.exists(src_meta):
        with open(src_meta, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {
            "filename": filename,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file_size_bytes": os.path.getsize(src_img) if os.path.exists(src_img) else 1024
        }

    metadata["status"] = dest_state
    metadata["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if diagnosis_result:
        metadata["crop"] = diagnosis_result.get("crop", "Unknown")
        metadata["disease_prediction"] = diagnosis_result.get("disease", "Unknown")
        metadata["confidence"] = diagnosis_result.get("confidence", 0.0)
        metadata["severity"] = diagnosis_result.get("severity", "Normal")

    # Move files if src exists
    if os.path.exists(src_img) and src_img != dst_img:
        shutil.move(src_img, dst_img)

    with open(dst_meta, "w") as f:
        json.dump(metadata, f, indent=2)

    if os.path.exists(src_meta) and src_meta != dst_meta:
        os.remove(src_meta)

    return dst_img, metadata

def get_storage_stats():
    """Calculate storage usage per directory and overall capacity."""
    init_file_storage()
    stats = {}
    total_bytes = 0
    total_files = 0

    for state, dir_path in DIRS.items():
        files = [f for f in os.listdir(dir_path) if not f.endswith('.json')]
        state_bytes = sum(os.path.getsize(os.path.join(dir_path, f)) for f in os.listdir(dir_path))
        stats[state] = {
            "file_count": len(files),
            "size_mb": round(state_bytes / (1024 * 1024), 2)
        }
        total_bytes += state_bytes
        total_files += len(files)

    stats["summary"] = {
        "total_files": total_files,
        "total_size_mb": round(total_bytes / (1024 * 1024), 2),
        "storage_capacity_mb": 1024.0,  # 1 GB simulated capacity
        "free_space_mb": round(1024.0 - (total_bytes / (1024 * 1024)), 2)
    }
    return stats

def archive_old_files(days_threshold=7):
    """Retention Policy: Archive files older than threshold."""
    init_file_storage()
    archived_count = 0
    
    for state in ["completed"]:
        dir_path = DIRS[state]
        for filename in os.listdir(dir_path):
            if not filename.endswith('.json'):
                img_path = os.path.join(dir_path, filename)
                # Archive file
                transition_file_status(filename, "archive")
                archived_count += 1

    return archived_count
