"""
Edge-to-Cloud Store-and-Forward Synchronization Manager.
Handles local storage when offline and auto-syncing when internet becomes available.
"""

import time
import random
from database.db import get_connection
from storage.file_manager import get_storage_stats

class CloudSyncManager:
    """Store-and-Forward Cloud Synchronization Manager."""
    def __init__(self):
        self.is_online = False
        self.bandwidth_kbps = 0.0  # 0 when offline
        self.sync_queue = []

    def set_connectivity(self, online: bool, bandwidth_kbps: float = 120.0):
        """Toggle online/offline network connectivity status."""
        self.is_online = online
        self.bandwidth_kbps = bandwidth_kbps if online else 0.0
        return {
            "is_online": self.is_online,
            "bandwidth_kbps": self.bandwidth_kbps,
            "mode": "Online Edge-to-Cloud Sync" if online else "Offline Local Store-and-Forward"
        }

    def trigger_sync(self):
        """Flush unsynced local data to simulated cloud storage if online."""
        if not self.is_online or self.bandwidth_kbps <= 0:
            return {
                "success": False,
                "message": "System is Offline. Images & diagnosis logs remain stored safely in local storage (/data/completed & /data/critical).",
                "synced_count": 0,
                "bandwidth_kbps": 0.0
            }

        conn = get_connection()
        cursor = conn.cursor()

        # Find unsynced leaf scans
        cursor.execute("SELECT scan_id, crop, disease, confidence, timestamp FROM leaf_scans WHERE status != 'synced';")
        rows = cursor.fetchall()
        count = len(rows)

        if count > 0:
            cursor.execute("UPDATE leaf_scans SET status = 'synced' WHERE status != 'synced';")
            cursor.execute("""
            INSERT INTO sync_log (timestamp, items_count, status, bandwidth_kbps)
            VALUES (datetime('now'), ?, 'SUCCESS', ?);
            """, (count, self.bandwidth_kbps))
            conn.commit()

        conn.close()

        storage_info = get_storage_stats()

        return {
            "success": True,
            "message": f"Successfully synchronized {count} records and images to AWS S3/Cloud Storage.",
            "synced_count": count,
            "bandwidth_kbps": self.bandwidth_kbps,
            "local_storage_stats": storage_info["summary"]
        }

sync_manager = CloudSyncManager()
