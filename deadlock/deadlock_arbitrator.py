"""
Deadlock & Resource Arbitration Module — OS Simulation.
Implements Banker's Algorithm (Safety & Resource Request) for resource allocation graph (RAG) arbitration.
Resources: CPU Cores, RAM Blocks, ML Engine, Network Channel.
"""

import numpy as np

RESOURCES = ["CPU", "RAM", "ML_Engine", "Network"]

class BankersAlgorithm:
    """Banker's Algorithm for Deadlock Avoidance and Safe State Verification."""
    def __init__(self, available, max_matrix, allocation):
        self.available = np.array(available, dtype=int)
        self.max_matrix = np.array(max_matrix, dtype=int)
        self.allocation = np.array(allocation, dtype=int)
        self.need = self.max_matrix - self.allocation
        self.num_processes = self.max_matrix.shape[0]

    def is_safe_state(self):
        """Check if the system is currently in a safe state and return the safe execution sequence."""
        work = np.copy(self.available)
        finish = np.zeros(self.num_processes, dtype=bool)
        safe_sequence = []

        while len(safe_sequence) < self.num_processes:
            found = False
            for i in range(self.num_processes):
                if not finish[i] and np.all(self.need[i] <= work):
                    work += self.allocation[i]
                    finish[i] = True
                    safe_sequence.append(i)
                    found = True
                    break
            
            if not found:
                break

        is_safe = (len(safe_sequence) == self.num_processes)
        return is_safe, safe_sequence

    def request_resources(self, process_id, request_vector):
        """Request resources for process_id and grant if state remains safe."""
        request = np.array(request_vector, dtype=int)
        
        # Condition 1: Request <= Need
        if np.any(request > self.need[process_id]):
            return False, "Error: Process exceeded its declared maximum claim.", []

        # Condition 2: Request <= Available
        if np.any(request > self.available):
            return False, "Resources unavailable. Process must wait (prevented deadlock/starvation).", []

        # Pretend to allocate resources
        self.available -= request
        self.allocation[process_id] += request
        self.need[process_id] -= request

        # Evaluate safety
        is_safe, safe_seq = self.is_safe_state()

        if is_safe:
            return True, f"Resource request granted for Task T{process_id}. System remains in Safe State.", safe_seq
        else:
            # Rollback allocation
            self.available += request
            self.allocation[process_id] -= request
            self.need[process_id] += request
            return False, f"Deadlock Risk Detected! Request denied for Task T{process_id}. System rolled back to safe state.", []


def simulate_deadlock_scenario(scenario_type="safe"):
    """Simulate multi-device resource contention scenarios (Safe vs Unsafe vs Prevented)."""

    processes = ["Camera 1 (Disease Scan)", "Camera 2 (Disease Scan)", "Drone (20 Images)", "Farmer Query NLP"]
    
    # Available System Resources [CPU Cores, RAM (x100MB), ML Engine, Network Channel]
    if scenario_type == "deadlock_risk":
        available = [1, 1, 0, 1]
    else:
        available = [3, 3, 2, 2]

    # Max matrix declared by tasks
    max_matrix = [
        [2, 2, 1, 1],  # Camera 1
        [2, 1, 1, 1],  # Camera 2
        [3, 3, 1, 2],  # Drone
        [1, 2, 0, 1]   # Farmer Query
    ]

    # Allocation matrix
    allocation = [
        [1, 1, 1, 0],  # Camera 1 holds ML Engine
        [1, 1, 0, 1],  # Camera 2 holds Network
        [1, 1, 1, 0],  # Drone holds ML Engine
        [0, 1, 0, 0]   # Query holds RAM
    ]

    banker = BankersAlgorithm(available, max_matrix, allocation)
    is_safe, safe_seq = banker.is_safe_state()

    # Simulate incoming request from Camera 2 for ML Engine [0, 0, 1, 0]
    granted, msg, seq = banker.request_resources(1, [0, 0, 1, 0])

    return {
        "status": "Safe State" if is_safe else "Unsafe / Deadlock Risk",
        "is_safe": is_safe,
        "safe_sequence": [processes[i] for i in safe_seq] if is_safe else [],
        "resources": RESOURCES,
        "available": banker.available.tolist(),
        "allocation_matrix": [
            {"process": processes[i], "alloc": banker.allocation[i].tolist()} for i in range(len(processes))
        ],
        "need_matrix": [
            {"process": processes[i], "need": banker.need[i].tolist()} for i in range(len(processes))
        ],
        "arbitration_result": {
            "request_task": processes[1],
            "request_vector": [0, 0, 1, 0],
            "granted": granted,
            "message": msg
        }
    }
