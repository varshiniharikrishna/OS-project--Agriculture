"""
Deadlock & Resource Arbitration Module — OS Simulation.
Implements Banker's Algorithm (Safety & Resource Request) for resource allocation graph (RAG) arbitration.
Resources: CPU Cores, RAM Blocks, ML Engine, Network Channel.
Implemented in Pure Python (zero external C-extension dependency) for maximum cross-platform resilience.
"""

RESOURCES = ["CPU", "RAM", "ML_Engine", "Network"]

class BankersAlgorithm:
    """Banker's Algorithm for Deadlock Avoidance and Safe State Verification."""
    def __init__(self, available, max_matrix, allocation):
        self.available = [int(x) for x in available]
        self.max_matrix = [[int(x) for x in row] for row in max_matrix]
        self.allocation = [[int(x) for x in row] for row in allocation]
        self.num_processes = len(self.max_matrix)
        self.num_resources = len(self.available)
        self.need = [
            [self.max_matrix[i][j] - self.allocation[i][j] for j in range(self.num_resources)]
            for i in range(self.num_processes)
        ]

    def is_safe_state(self):
        """Check if the system is currently in a safe state and return the safe execution sequence."""
        work = list(self.available)
        finish = [False] * self.num_processes
        safe_sequence = []

        while len(safe_sequence) < self.num_processes:
            found = False
            for i in range(self.num_processes):
                if not finish[i] and all(self.need[i][j] <= work[j] for j in range(self.num_resources)):
                    for j in range(self.num_resources):
                        work[j] += self.allocation[i][j]
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
        req = [int(x) for x in request_vector]
        
        # Condition 1: Request <= Need
        if any(req[j] > self.need[process_id][j] for j in range(self.num_resources)):
            return False, "Error: Process exceeded its declared maximum claim.", []

        # Condition 2: Request <= Available
        if any(req[j] > self.available[j] for j in range(self.num_resources)):
            return False, "Resources unavailable. Process must wait (prevented deadlock/starvation).", []

        # Pretend to allocate resources
        for j in range(self.num_resources):
            self.available[j] -= req[j]
            self.allocation[process_id][j] += req[j]
            self.need[process_id][j] -= req[j]

        # Evaluate safety
        is_safe, safe_seq = self.is_safe_state()

        if is_safe:
            return True, f"Resource request granted for Task T{process_id}. System remains in Safe State.", safe_seq
        else:
            # Rollback allocation
            for j in range(self.num_resources):
                self.available[j] += req[j]
                self.allocation[process_id][j] -= req[j]
                self.need[process_id][j] += req[j]
            return False, f"Deadlock Risk Detected! Request denied for Task T{process_id}. System rolled back to safe state.", []


def simulate_deadlock_scenario(scenario_type="safe"):
    """Simulate multi-device resource contention scenarios (Safe vs Unsafe vs Prevented)."""

    processes = ["Leaf Disease Scanner", "Soil Sensor Node", "Cloud Sync Worker", "Farmer Offline PDF Query"]
    
    # Available System Resources [CPU Cores, RAM (x100MB), ML Engine, Network Channel]
    if scenario_type == "deadlock_risk":
        available = [1, 1, 0, 1]
    else:
        available = [3, 3, 2, 2]

    # Max matrix declared by tasks
    max_matrix = [
        [2, 2, 1, 1],  # Leaf Disease Scanner
        [2, 1, 1, 1],  # Soil Sensor Node
        [3, 3, 1, 2],  # Cloud Sync Worker
        [1, 2, 0, 1]   # Farmer Offline PDF Query
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

    # Simulate incoming request from Soil Sensor Node for ML Engine [0, 0, 1, 0]
    granted, msg, seq = banker.request_resources(1, [0, 0, 1, 0])

    return {
        "status": "Safe State" if is_safe else "Unsafe / Deadlock Risk",
        "is_safe": is_safe,
        "safe_sequence": [processes[i] for i in safe_seq] if is_safe else [],
        "resources": RESOURCES,
        "available": list(banker.available),
        "allocation_matrix": [
            {"process": processes[i], "alloc": list(banker.allocation[i])} for i in range(len(processes))
        ],
        "need_matrix": [
            {"process": processes[i], "need": list(banker.need[i])} for i in range(len(processes))
        ],
        "arbitration_result": {
            "request_task": processes[1],
            "request_vector": [0, 0, 1, 0],
            "granted": granted,
            "message": msg
        }
    }
