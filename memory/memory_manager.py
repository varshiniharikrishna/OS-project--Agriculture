"""
Memory Management Module — OS Simulation.
Simulates RAM allocation, model loading/unloading, and LRU cache eviction under memory constraints.
"""

from collections import OrderedDict
import time

TOTAL_RAM_MB = 512.0  # Simulated Edge Device RAM Capacity (e.g. 512 MB)

MODEL_MEMORY_FOOTPRINTS = {
    "EfficientNet-B0": 180.0,
    "ResNet-50": 340.0,
    "Image_Compressor": 45.0,
    "Sensor_Processor": 25.0,
    "Cloud_Sync_Worker": 35.0,
    "Farmer_Assistance_NLP": 95.0
}

class MemoryManager:
    """Simulated RAM Manager with LRU eviction strategy."""
    def __init__(self, total_ram_mb=TOTAL_RAM_MB):
        self.total_ram_mb = total_ram_mb
        self.loaded_models = OrderedDict()  # LRU cache: model_name -> footprint_mb
        self.eviction_history = []

    @property
    def used_ram_mb(self):
        return sum(self.loaded_models.values())

    @property
    def free_ram_mb(self):
        return max(0.0, self.total_ram_mb - self.used_ram_mb)

    def allocate(self, model_name):
        """Allocate RAM for a task/model, performing LRU eviction if capacity is exceeded."""
        footprint = MODEL_MEMORY_FOOTPRINTS.get(model_name, 100.0)
        
        # If already loaded, move to end (mark as recently used)
        if model_name in self.loaded_models:
            self.loaded_models.move_to_end(model_name)
            return {
                "status": "already_loaded",
                "model": model_name,
                "used_ram_mb": self.used_ram_mb,
                "free_ram_mb": self.free_ram_mb,
                "evicted": []
            }

        evicted_list = []
        # Check if eviction is necessary
        while self.free_ram_mb < footprint and len(self.loaded_models) > 0:
            # Evict LRU (first item in OrderedDict)
            lru_model, lru_size = self.loaded_models.popitem(last=False)
            evicted_list.append(lru_model)
            timestamp = time.strftime("%H:%M:%S")
            self.eviction_history.append({
                "timestamp": timestamp,
                "evicted_model": lru_model,
                "freed_mb": lru_size,
                "reason": f"Insufficient RAM for loading {model_name}"
            })

        # Load new model
        self.loaded_models[model_name] = footprint
        return {
            "status": "allocated",
            "model": model_name,
            "allocated_mb": footprint,
            "used_ram_mb": self.used_ram_mb,
            "free_ram_mb": self.free_ram_mb,
            "evicted": evicted_list
        }

    def unload(self, model_name):
        """Manually unload a model from RAM."""
        if model_name in self.loaded_models:
            freed = self.loaded_models.pop(model_name)
            return {"status": "unloaded", "model": model_name, "freed_mb": freed}
        return {"status": "not_found", "model": model_name}

    def get_state(self):
        """Get current RAM utilization snapshot."""
        return {
            "total_ram_mb": self.total_ram_mb,
            "used_ram_mb": round(self.used_ram_mb, 1),
            "free_ram_mb": round(self.free_ram_mb, 1),
            "utilization_pct": round((self.used_ram_mb / self.total_ram_mb) * 100, 1),
            "active_models": [
                {"name": name, "ram_mb": size} for name, size in self.loaded_models.items()
            ],
            "recent_evictions": self.eviction_history[-5:]
        }

# Global MemoryManager singleton
memory_manager = MemoryManager()
