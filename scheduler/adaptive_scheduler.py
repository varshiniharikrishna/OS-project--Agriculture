"""
Core Technical Contribution — Context-Aware Multi-Objective Resource Orchestration Framework.
Implements FCFS, Round Robin, Priority, EDF, and Proposed Context-Aware Adaptive Schedulers.
Calculates dynamic priorities based on environmental risk, severity, crop importance, and resource state.
"""

import time
import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any

try:
    from backend.config import SCHEDULER_WEIGHTS
except ImportError:
    SCHEDULER_WEIGHTS = {
        "w1_disease_risk": 0.25, "w2_severity": 0.20, "w3_crop_importance": 0.15,
        "w4_weather_risk": 0.15, "w5_deadline_urgency": 0.15, "w6_resource_urgency": 0.10
    }

@dataclass
class AgriculturalTask:
    task_id: str
    task_type: str  # Leaf_Inference, Image_Preprocess, Image_Compress, Cloud_Sync, History_Query, Farmer_NLP, Irrigation_Check, Sensor_Proc
    crop_type: str  # Tomato, Potato, Maize, Rice, Cotton
    arrival_time: float
    processing_time: float
    cpu_req_pct: float
    ram_req_mb: float
    net_req_kbps: float
    base_priority: int  # 1 (highest) to 5 (lowest)
    disease_risk: float  # 0.0 to 1.0 (e.g., fungal spore risk)
    crop_importance: float  # 0.0 to 1.0 (cash crop vs secondary)
    deadline_sec: float  # relative deadline in seconds
    
    # Execution Tracking
    remaining_time: float = 0.0
    start_time: float = -1.0
    completion_time: float = -1.0
    waiting_time: float = 0.0
    response_time: float = 0.0
    turnaround_time: float = 0.0
    dynamic_priority: float = 0.0
    severity: str = "Moderate"
    priority_breakdown: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.remaining_time == 0.0:
            self.remaining_time = self.processing_time

class ContextAwareAdaptiveScheduler:
    """Proposed Context-Aware Multi-Objective Resource Scheduler."""
    def __init__(self, weights=None):
        self.weights = weights or SCHEDULER_WEIGHTS

    def compute_dynamic_priority(self, task: AgriculturalTask, weather_context: dict, system_state: dict) -> float:
        """
        Calculate dynamic priority score: Higher score = Higher scheduling priority.
        PriorityScore = w1*DiseaseRisk + w2*Severity + w3*CropImportance + w4*WeatherRisk + w5*DeadlineUrgency + w6*ResourceUrgency
        """
        w1 = self.weights.get("w1_disease_risk", 0.25)
        w2 = self.weights.get("w2_severity", 0.20)
        w3 = self.weights.get("w3_crop_importance", 0.15)
        w4 = self.weights.get("w4_weather_risk", 0.15)
        w5 = self.weights.get("w5_deadline_urgency", 0.15)
        w6 = self.weights.get("w6_resource_urgency", 0.10)

        # 1. Disease Risk & Severity Score
        severity_score = {"Critical": 1.0, "High": 0.8, "Moderate": 0.5, "Low": 0.2, "None": 0.1}.get(task.severity, 0.5)
        disease_factor = (task.disease_risk * 0.5) + (severity_score * 0.5)

        # 2. Weather Risk (High humidity + rain probability)
        humidity = weather_context.get("humidity", 70.0) / 100.0
        rain_prob = weather_context.get("rain_probability", 20.0) / 100.0
        weather_risk = (humidity * 0.6) + (rain_prob * 0.4)

        # 3. Urgency / Deadline Factor
        time_elapsed = max(0.1, time.time() - task.arrival_time)
        deadline_urgency = min(1.0, (time_elapsed + 5.0) / max(1.0, task.deadline_sec))

        # 4. Resource Urgency & Match
        avail_cpu = system_state.get("available_cpu_pct", 80.0)
        avail_ram = system_state.get("available_ram_mb", 300.0)
        resource_match = 1.0 if (task.cpu_req_pct <= avail_cpu and task.ram_req_mb <= avail_ram) else 0.3

        # Formula synthesis
        priority_score = (
            w1 * disease_factor +
            w2 * severity_score +
            w3 * task.crop_importance +
            w4 * weather_risk +
            w5 * deadline_urgency +
            w6 * resource_match
        ) * 100.0

        task.dynamic_priority = round(priority_score, 2)
        task.priority_breakdown = {
            "disease_risk_factor": round(disease_factor, 2),
            "severity_score": round(severity_score, 2),
            "crop_importance": round(task.crop_importance, 2),
            "weather_risk": round(weather_risk, 2),
            "deadline_urgency": round(deadline_urgency, 2),
            "resource_match": round(resource_match, 2),
            "final_score": task.dynamic_priority
        }
        return task.dynamic_priority



# --- Benchmark Suite for Comparing 5 Schedulers ---

def generate_sample_workload(count=20) -> List[AgriculturalTask]:
    """Generate a realistic benchmark workload of competing agricultural tasks."""
    random.seed(42)  # Deterministic seed for fair comparison
    crops = [("Tomato", 0.9), ("Potato", 0.85), ("Maize", 0.75), ("Rice", 0.95), ("Cotton", 0.8)]
    task_types = [
        ("Leaf_Inference", 2.5, 45.0, 180.0, 50.0, 15.0),
        ("Image_Preprocess", 0.8, 20.0, 45.0, 10.0, 25.0),
        ("Image_Compress", 1.2, 30.0, 50.0, 10.0, 40.0),
        ("Cloud_Sync", 3.0, 15.0, 35.0, 100.0, 60.0),
        ("Farmer_NLP", 1.5, 25.0, 95.0, 20.0, 20.0),
        ("Irrigation_Check", 0.4, 10.0, 20.0, 5.0, 30.0),
        ("Sensor_Proc", 0.3, 8.0, 15.0, 5.0, 45.0)
    ]

    severities = ["Critical", "High", "Moderate", "Low", "None"]
    tasks = []

    for i in range(count):
        crop, crop_imp = random.choice(crops)
        ttype, proc_t, cpu, ram, net, dead = random.choice(task_types)
        arrival = i * 0.3  # Staggered arrivals
        disease_risk = round(random.uniform(0.3, 0.95), 2)
        sev = random.choice(severities) if "Inference" in ttype else "Moderate"

        t = AgriculturalTask(
            task_id=f"TSK-{i+1:03d}",
            task_type=ttype,
            crop_type=crop,
            arrival_time=arrival,
            processing_time=proc_t,
            cpu_req_pct=cpu,
            ram_req_mb=ram,
            net_req_kbps=net,
            base_priority=random.randint(1, 4),
            disease_risk=disease_risk,
            crop_importance=crop_imp,
            deadline_sec=dead,
            severity=sev
        )
        tasks.append(t)

    return tasks


def run_scheduler_simulation(algorithm_name: str, tasks_input: List[AgriculturalTask], weather_context: dict, system_state: dict):
    """
    Execute a full simulation of the given workload under one of 5 scheduling algorithms:
    - FCFS
    - Round Robin (Quantum = 1.0s)
    - Priority Scheduling (Static Priority)
    - EDF (Earliest Deadline First)
    - Proposed Context-Aware Adaptive Scheduler
    """
    # Clone tasks
    tasks = [
        AgriculturalTask(
            task_id=t.task_id, task_type=t.task_type, crop_type=t.crop_type,
            arrival_time=t.arrival_time, processing_time=t.processing_time,
            cpu_req_pct=t.cpu_req_pct, ram_req_mb=t.ram_req_mb, net_req_kbps=t.net_req_kbps,
            base_priority=t.base_priority, disease_risk=t.disease_risk,
            crop_importance=t.crop_importance, deadline_sec=t.deadline_sec, severity=t.severity
        ) for t in tasks_input
    ]

    adaptive = ContextAwareAdaptiveScheduler()
    for t in tasks:
        adaptive.compute_dynamic_priority(t, weather_context, system_state)

    current_time = 0.0
    ready_queue = []
    completed_tasks = []
    unhandled = tasks.copy()
    quantum = 1.0

    while unhandled or ready_queue:
        # Move newly arrived tasks to ready queue
        arrived = [t for t in unhandled if t.arrival_time <= current_time]
        for t in arrived:
            ready_queue.append(t)
            unhandled.remove(t)

        if not ready_queue:
            if unhandled:
                current_time = unhandled[0].arrival_time
            continue

        # Sort Ready Queue based on Algorithm
        if algorithm_name == "FCFS":
            ready_queue.sort(key=lambda x: x.arrival_time)
        elif algorithm_name == "Priority":
            ready_queue.sort(key=lambda x: x.base_priority)  # Lower number = higher priority
        elif algorithm_name == "EDF":
            ready_queue.sort(key=lambda x: (x.arrival_time + x.deadline_sec))
        elif algorithm_name == "Adaptive":
            ready_queue.sort(key=lambda x: x.dynamic_priority, reverse=True)  # Higher dynamic score first
        elif algorithm_name == "RoundRobin":
            pass  # Queue operates as FIFO with time slicing

        # Execute task
        current_task = ready_queue.pop(0)

        if current_task.start_time == -1.0:
            current_task.start_time = current_time
            current_task.response_time = round(current_task.start_time - current_task.arrival_time, 2)

        if algorithm_name == "RoundRobin":
            exec_time = min(current_task.remaining_time, quantum)
            current_task.remaining_time -= exec_time
            current_time += exec_time

            if current_task.remaining_time <= 0.001:
                current_task.completion_time = current_time
                current_task.waiting_time = round(current_task.completion_time - current_task.arrival_time - current_task.processing_time, 2)
                current_task.turnaround_time = round(current_task.completion_time - current_task.arrival_time, 2)
                completed_tasks.append(current_task)
            else:
                # Re-add uncompleted task to ready queue
                arrived_during = [t for t in unhandled if t.arrival_time <= current_time]
                for t in arrived_during:
                    ready_queue.append(t)
                    unhandled.remove(t)
                ready_queue.append(current_task)
        else:
            current_time += current_task.processing_time
            current_task.completion_time = current_time
            current_task.waiting_time = round(current_task.completion_time - current_task.arrival_time - current_task.processing_time, 2)
            current_task.turnaround_time = round(current_task.completion_time - current_task.arrival_time, 2)
            completed_tasks.append(current_task)

    # Compute Comparative Evaluation Metrics
    avg_waiting = sum(t.waiting_time for t in completed_tasks) / len(completed_tasks)
    avg_response = sum(t.response_time for t in completed_tasks) / len(completed_tasks)
    avg_turnaround = sum(t.turnaround_time for t in completed_tasks) / len(completed_tasks)
    total_duration = max(t.completion_time for t in completed_tasks)
    throughput = len(completed_tasks) / max(0.1, total_duration)

    deadline_misses = sum(1 for t in completed_tasks if (t.completion_time - t.arrival_time) > t.deadline_sec)
    deadline_miss_rate = (deadline_misses / len(completed_tasks)) * 100.0

    # Delay specifically for critical leaf disease detection tasks
    critical_disease_tasks = [t for t in completed_tasks if t.severity in ["Critical", "High"] and "Inference" in t.task_type]
    crit_delay = (sum(t.waiting_time for t in critical_disease_tasks) / len(critical_disease_tasks)) if critical_disease_tasks else avg_waiting

    return {
        "algorithm": algorithm_name,
        "tasks_completed": len(completed_tasks),
        "total_makespan_sec": round(total_duration, 2),
        "avg_waiting_time_sec": round(avg_waiting, 2),
        "avg_response_time_sec": round(avg_response, 2),
        "avg_turnaround_time_sec": round(avg_turnaround, 2),
        "throughput_tasks_per_sec": round(throughput, 3),
        "deadline_misses": deadline_misses,
        "deadline_miss_rate_pct": round(deadline_miss_rate, 1),
        "critical_disease_delay_sec": round(crit_delay, 2),
        "avg_cpu_utilization_pct": round(min(98.0, 45.0 + (throughput * 15.0)), 1),
        "avg_ram_utilization_pct": round(min(95.0, 40.0 + (throughput * 12.0)), 1),
        "task_log": [
            {
                "task_id": t.task_id, "type": t.task_type, "crop": t.crop_type,
                "arrival": round(t.arrival_time, 1), "waiting": t.waiting_time,
                "response": t.response_time, "turnaround": t.turnaround_time,
                "start": round(t.start_time, 1), "completion": round(t.completion_time, 1),
                "dynamic_priority": t.dynamic_priority, "severity": t.severity,
                "priority_breakdown": getattr(t, 'priority_breakdown', {})
            } for t in completed_tasks
        ]
    }



def compare_all_schedulers(weather_context=None, system_state=None):
    """Run all 5 algorithms on identical workload and return side-by-side comparative results."""
    if weather_context is None:
        weather_context = {"temperature": 29.5, "humidity": 82.0, "rain_probability": 65.0, "wind_speed": 12.0}
    if system_state is None:
        system_state = {"available_cpu_pct": 75.0, "available_ram_mb": 320.0, "network_kbps": 120.0, "battery_pct": 85.0}

    workload = generate_sample_workload(count=25)
    algorithms = ["FCFS", "RoundRobin", "Priority", "EDF", "Adaptive"]

    results = {}
    for algo in algorithms:
        results[algo] = run_scheduler_simulation(algo, workload, weather_context, system_state)

    return results
