"""
Context-Aware Multi-Objective Resource Orchestration Scheduler.
──────────────────────────────────────────────────────────────
Core OS Contribution.

Implements five scheduling algorithms on a common workload:
  1. FCFS                 – First-Come First-Served
  2. RoundRobin           – Round Robin (quantum = 1.0 s)
  3. Priority             – Static priority (1 = highest)
  4. EDF                  – Earliest Deadline First
  5. Adaptive (proposed)  – Context-Aware Multi-Objective Scheduler

Priority formula (Adaptive):
  S_dynamic = 100 × (0.25·D + 0.20·S + 0.15·C + 0.15·W + 0.15·U + 0.10·R)

Where (all factors ∈ [0, 1]):
  D  = disease_risk          (input field, e.g. fungal spore density proxy)
  S  = severity_score        (Critical=1.0, High=0.8, Moderate=0.5, Low=0.2, None=0.1)
  C  = crop_importance       (economic weight, 0–1)
  W  = weather_risk          = 0.60·(humidity/100) + 0.40·(rain_prob/100)
  U  = deadline_urgency      = clamp(elapsed / deadline, 0, 1)
  R  = resource_compatibility = 1.0 if task fits available CPU & RAM, else 0.3

NOTE: D and S are intentionally separate.
  D measures environmental/epidemiological disease pressure (external context).
  S measures the observed severity of the already-detected leaf disease (outcome severity).
  Together they ensure both "risk of future spread" and "urgency of current case" are weighted.
"""

import time
import copy
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ── Task data structure ───────────────────────────────────────────────────────

@dataclass
class AgriculturalTask:
    task_id: str
    task_type: str          # Leaf_Inference | Image_Preprocess | Cloud_Sync | Farmer_NLP | …
    crop_type: str          # Tomato | Potato | Maize | Rice | Cotton | …
    arrival_time: float     # seconds from epoch of simulation start
    processing_time: float  # CPU burst in seconds
    cpu_req_pct: float      # CPU percentage required (0–100)
    ram_req_mb: float       # RAM required in MB
    net_req_kbps: float     # Network required in Kbps
    base_priority: int      # Static priority 1 (highest) – 4 (lowest)
    disease_risk: float     # Environmental/epidemiological risk factor ∈ [0, 1]
    crop_importance: float  # Economic crop weight ∈ [0, 1]
    deadline_sec: float     # Time budget for completion (seconds after arrival)
    severity: str = "Moderate"  # Observed disease severity label

    # Execution tracking (filled during simulation)
    remaining_time: float = 0.0
    start_time: float = -1.0
    completion_time: float = -1.0
    waiting_time: float = 0.0
    response_time: float = 0.0
    dynamic_priority: float = 0.0
    turnaround_time: float = 0.0

    def __post_init__(self):
        if self.remaining_time == 0.0:
            self.remaining_time = self.processing_time


# ── Adaptive priority engine ──────────────────────────────────────────────────

class ContextAwareAdaptiveScheduler:
    """
    Computes S_dynamic for every task using agricultural + resource context.
    Weights sum to 1.0 (0.25+0.20+0.15+0.15+0.15+0.10 = 1.00).
    """

    W_DISEASE   = 0.25   # D – environmental disease risk
    W_SEVERITY  = 0.20   # S – observed leaf severity
    W_CROP      = 0.15   # C – crop economic importance
    W_WEATHER   = 0.15   # W – weather-derived risk
    W_DEADLINE  = 0.15   # U – deadline urgency
    W_RESOURCE  = 0.10   # R – resource compatibility

    SEVERITY_MAP = {
        "Critical": 1.0, "High": 0.8, "Moderate": 0.5, "Low": 0.2, "None": 0.1
    }

    def compute(
        self,
        task: AgriculturalTask,
        weather: dict,
        system: dict,
        current_time: float = 0.0,
    ) -> float:
        """
        Calculate S_dynamic ∈ [0, 100].

        Parameters
        ----------
        task         : AgriculturalTask instance.
        weather      : dict with 'humidity' (0–100) and 'rain_probability' (0–100).
        system       : dict with 'available_cpu_pct' and 'available_ram_mb'.
        current_time : wall-clock time of scheduling decision.
        """
        # D – disease risk (already ∈ [0,1])
        D = float(task.disease_risk)

        # S – severity
        S = self.SEVERITY_MAP.get(task.severity, 0.5)

        # C – crop importance (already ∈ [0,1])
        C = float(task.crop_importance)

        # W – weather risk: W = 0.60·(humidity/100) + 0.40·(rain_prob/100)
        humidity  = float(weather.get("humidity", 70.0))
        rain_prob = float(weather.get("rain_probability", 20.0))
        W = 0.60 * (humidity / 100.0) + 0.40 * (rain_prob / 100.0)
        W = min(1.0, max(0.0, W))

        # U – deadline urgency: clamp(elapsed / deadline, 0, 1)
        elapsed = max(0.0, current_time - task.arrival_time)
        U = min(1.0, elapsed / max(0.1, task.deadline_sec))

        # R – resource compatibility
        avail_cpu = float(system.get("available_cpu_pct", 80.0))
        avail_ram = float(system.get("available_ram_mb", 300.0))
        R = 1.0 if (task.cpu_req_pct <= avail_cpu and task.ram_req_mb <= avail_ram) else 0.3

        # Assemble score
        score = 100.0 * (
            self.W_DISEASE  * D +
            self.W_SEVERITY * S +
            self.W_CROP     * C +
            self.W_WEATHER  * W +
            self.W_DEADLINE * U +
            self.W_RESOURCE * R
        )

        task.dynamic_priority = round(score, 3)
        return task.dynamic_priority


# ── Simulator ─────────────────────────────────────────────────────────────────

_ADAPTIVE = ContextAwareAdaptiveScheduler()
QUANTUM = 1.0   # Round Robin time quantum (seconds)


def _clone_tasks(src: List[AgriculturalTask]) -> List[AgriculturalTask]:
    return [copy.deepcopy(t) for t in src]


def run_scheduler_simulation(
    algorithm_name: str,
    tasks_input: List[AgriculturalTask],
    weather_context: dict,
    system_state: dict,
) -> dict:
    """
    Simulate one of the five schedulers on tasks_input.
    Returns a metrics dict suitable for frontend charts and comparison tables.
    """
    tasks = _clone_tasks(tasks_input)

    # Pre-compute adaptive priorities (used even in baseline runs for display)
    current_wall = 0.0
    for t in tasks:
        _ADAPTIVE.compute(t, weather_context, system_state, current_time=t.arrival_time)

    # Sort input by arrival so simulation starts chronologically
    tasks.sort(key=lambda t: t.arrival_time)

    current_time = 0.0
    unhandled = list(tasks)
    ready_queue: List[AgriculturalTask] = []
    completed: List[AgriculturalTask] = []

    def admit_arrived():
        newly = [t for t in unhandled if t.arrival_time <= current_time]
        for t in newly:
            unhandled.remove(t)
            ready_queue.append(t)

    while unhandled or ready_queue:
        admit_arrived()

        if not ready_queue:
            if unhandled:
                current_time = unhandled[0].arrival_time
                continue
            break

        # Sort ready queue according to algorithm
        if algorithm_name == "FCFS":
            ready_queue.sort(key=lambda t: t.arrival_time)
        elif algorithm_name == "Priority":
            ready_queue.sort(key=lambda t: t.base_priority)   # 1 = best
        elif algorithm_name == "EDF":
            ready_queue.sort(key=lambda t: t.arrival_time + t.deadline_sec)
        elif algorithm_name == "Adaptive":
            # Re-score with current wall time so U factor is live
            for t in ready_queue:
                _ADAPTIVE.compute(t, weather_context, system_state, current_time=current_time)
            ready_queue.sort(key=lambda t: t.dynamic_priority, reverse=True)
        # FCFS and RoundRobin: FIFO order (no additional sort)

        task = ready_queue.pop(0)

        if task.start_time == -1.0:
            task.start_time = current_time
            task.response_time = round(current_time - task.arrival_time, 4)

        if algorithm_name == "RoundRobin":
            exec_time = min(task.remaining_time, QUANTUM)
            task.remaining_time -= exec_time
            current_time += exec_time

            admit_arrived()

            if task.remaining_time <= 1e-6:
                task.completion_time = current_time
                task.waiting_time = round(
                    task.completion_time - task.arrival_time - task.processing_time, 4
                )
                task.turnaround_time = round(task.completion_time - task.arrival_time, 4)
                completed.append(task)
            else:
                ready_queue.append(task)   # re-queue
        else:
            current_time += task.processing_time
            task.completion_time = current_time
            task.waiting_time = round(
                task.completion_time - task.arrival_time - task.processing_time, 4
            )
            task.turnaround_time = round(task.completion_time - task.arrival_time, 4)
            completed.append(task)

    if not completed:
        return {"algorithm": algorithm_name, "error": "No tasks completed"}

    n = len(completed)
    total_makespan = max(t.completion_time for t in completed)
    avg_wait       = sum(t.waiting_time for t in completed) / n
    avg_response   = sum(t.response_time for t in completed) / n
    avg_turnaround = sum(t.turnaround_time for t in completed) / n
    throughput     = n / max(0.01, total_makespan)

    deadline_misses = sum(
        1 for t in completed
        if (t.completion_time - t.arrival_time) > t.deadline_sec
    )
    miss_rate = (deadline_misses / n) * 100.0

    # Critical disease detection tasks
    crit = [t for t in completed if t.severity in ("Critical", "High") and "Inference" in t.task_type]
    crit_delay = (sum(t.waiting_time for t in crit) / len(crit)) if crit else avg_wait

    # CPU utilisation proxy (from actual resource requirements)
    cpu_util = min(98.0, sum(t.cpu_req_pct * t.processing_time for t in completed) / max(0.01, total_makespan))
    ram_util = min(98.0, sum(t.ram_req_mb for t in completed) / (n * 5.12))   # normalised to 512 MB

    return {
        "algorithm": algorithm_name,
        "tasks_completed": n,
        "total_makespan_sec": round(total_makespan, 3),
        "avg_waiting_time_sec": round(max(0.0, avg_wait), 3),
        "avg_response_time_sec": round(max(0.0, avg_response), 3),
        "avg_turnaround_time_sec": round(avg_turnaround, 3),
        "throughput_tasks_per_sec": round(throughput, 4),
        "deadline_misses": deadline_misses,
        "deadline_miss_rate_pct": round(miss_rate, 1),
        "critical_disease_delay_sec": round(max(0.0, crit_delay), 3),
        "cpu_utilization_pct": round(cpu_util, 1),
        "ram_utilization_pct": round(ram_util, 1),
        "task_log": [
            {
                "task_id": t.task_id,
                "type": t.task_type,
                "crop": t.crop_type,
                "arrival": round(t.arrival_time, 2),
                "burst": round(t.processing_time, 2),
                "waiting": round(t.waiting_time, 2),
                "response": round(t.response_time, 2),
                "turnaround": round(t.turnaround_time, 2),
                "completion": round(t.completion_time, 2),
                "deadline": t.deadline_sec,
                "missed": (t.completion_time - t.arrival_time) > t.deadline_sec,
                "dynamic_priority": round(t.dynamic_priority, 2),
                "severity": t.severity,
                "base_priority": t.base_priority,
            }
            for t in completed[:12]
        ],
    }


# ── Public comparison runner ──────────────────────────────────────────────────

def generate_sample_workload(count: int = 30) -> List[AgriculturalTask]:
    """
    Generate benchmark workload using the real Google Cluster Trace v3 loader.
    Falls back to Pareto-Beta synthetic if trace data is absent.
    """
    try:
        from scheduler.workload_loader import generate_workload
        return generate_workload(count=count, seed=42)
    except Exception:
        pass

    # Minimal fallback so module is always importable
    random.seed(42)
    tasks = []
    for i in range(count):
        tasks.append(AgriculturalTask(
            task_id=f"TSK-FB-{i+1:03d}",
            task_type=random.choice(["Leaf_Inference", "Farmer_NLP", "Cloud_Sync"]),
            crop_type=random.choice(["Tomato", "Potato", "Rice"]),
            arrival_time=i * 0.3,
            processing_time=round(random.uniform(0.3, 3.0), 2),
            cpu_req_pct=round(random.uniform(10, 45), 1),
            ram_req_mb=round(random.uniform(20, 180), 1),
            net_req_kbps=round(random.uniform(5, 100), 1),
            base_priority=random.randint(1, 4),
            disease_risk=round(random.uniform(0.3, 0.95), 3),
            crop_importance=round(random.uniform(0.7, 0.95), 2),
            deadline_sec=random.choice([15, 20, 25, 30, 40, 60]),
            severity=random.choice(["Critical", "High", "Moderate", "Low", "None"]),
        ))
    return tasks


def compare_all_schedulers(
    weather_context: Optional[dict] = None,
    system_state: Optional[dict] = None,
    task_count: int = 30,
) -> dict:
    """
    Run all five schedulers on the same workload and return comparative results.
    """
    weather = weather_context or {
        "temperature": 29.5, "humidity": 82.0,
        "rain_probability": 65.0, "wind_speed": 12.0
    }
    system = system_state or {
        "available_cpu_pct": 75.0, "available_ram_mb": 320.0,
        "network_kbps": 120.0, "battery_pct": 85.0
    }

    workload = generate_sample_workload(count=task_count)
    algorithms = ["FCFS", "RoundRobin", "Priority", "EDF", "Adaptive"]

    results = {}
    for algo in algorithms:
        results[algo] = run_scheduler_simulation(algo, workload, weather, system)

    return results
