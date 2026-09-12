# Context-Aware Multi-Objective Resource Orchestration Framework for Real-Time Agricultural Edge Intelligence

**Authors:** Varshini Harikrishna (24BCE0780) & Naurin Shabeer (24BCE0772)

---

## Executive Summary

This application is a complete, working web application built for our Operating Systems project. The system combines **real PyTorch crop disease detection (EfficientNet-B0)** with an **OS-level resource orchestration framework** designed specifically for resource-constrained edge computing devices operating in offline or intermittently connected Indian agricultural environments.

---

## Key Features

1. **Crop Disease Detection (Core Feature)**:
   - **Primary Model**: EfficientNet-B0 (with ResNet-50 alternative).
   - **Dataset**: PlantVillage Dataset (38 crop and disease classes).
   - **Diagnosis Output**: Image preview, crop identified, disease predicted, confidence percentage, plant health status (Healthy / Diseased), severity badge, and non-hallucinated actionable recommendations from a controlled knowledge base.

2. **Core OS Contribution — Adaptive Resource Scheduler**:
   - Implements 5 scheduling algorithms:
     1. **FCFS** (First-Come, First-Served)
     2. **Round Robin** (Time Quantum = 1.0s)
     3. **Priority Scheduling** (Static Priority)
     4. **EDF** (Earliest Deadline First)
     5. **Proposed Context-Aware Adaptive Scheduler** (Novel dynamic priority formula)
   - **Dynamic Priority Formula**:
     $$\text{Priority} = w_1 \cdot \text{DiseaseRisk} + w_2 \cdot \text{Severity} + w_3 \cdot \text{CropImportance} + w_4 \cdot \text{WeatherRisk} + w_5 \cdot \frac{1}{\text{Deadline}} + w_6 \cdot \text{ResourceMatch}$$

3. **Resource & Memory Management**:
   - Simulates RAM capacity (512 MB) with **LRU (Least Recently Used)** model loading/unloading.
   - Monitors CPU, RAM, Storage, Network Bandwidth, and Battery status.

4. **Deadlock & Resource Arbitration**:
   - Implements **Banker's Algorithm** and Resource Allocation Graph (RAG) for CPU, RAM, ML Engine, and Network channel locks across multiple concurrent devices (Camera 1, Camera 2, Drone).

5. **File Management System (OS Objective)**:
   - 5-stage directory hierarchy: `/data/pending/`, `/data/processing/`, `/data/completed/`, `/data/critical/`, `/data/archive/`.
   - Metadata JSON sidecars, status state transitions, and retention policy auto-archiving.

6. **Multilingual Support**:
   - Dropdown switcher supporting **English**, **Tamil**, **Hindi**, **Telugu**, **Kannada**, **Malayalam**, **Bengali**, and **Marathi**.

7. **Farmer Assistance Panel**:
   - Integrated Q&A panel providing non-hallucinated guidance on watering, fertilizers, disease prevention, and livestock safety.

---

## Project Structure

```
agri_edge_os/
├── backend/
│   ├── app.py                # Main Flask Server & REST Endpoints
│   ├── config.py             # Configuration & Constants
│   └── sync_manager.py       # Store-and-Forward Edge-to-Cloud Sync
├── ml/
│   ├── efficientnet_b0.py    # PyTorch EfficientNet-B0 & ResNet-50 Architectures
│   ├── inference.py          # Preprocessing & Inference Engine
│   ├── train.py              # Fine-Tuning Training Pipeline
│   ├── knowledge_base.py     # Controlled Agricultural Knowledge Base
│   ├── models/               # Saved PyTorch Checkpoints
│   └── dataset/              # PlantVillage Dataset Upload Directory
├── scheduler/
│   └── adaptive_scheduler.py # FCFS, RR, Priority, EDF & Proposed Adaptive Schedulers
├── memory/
│   └── memory_manager.py     # RAM Manager with LRU Eviction Policy
├── storage/
│   └── file_manager.py       # 5-Stage File Storage Directory Manager
├── deadlock/
│   └── deadlock_arbitrator.py# Banker's Algorithm & RAG Deadlock Prevention
├── database/
│   └── db.py                 # SQLite DAO & Schema Initialization
├── translations/             # Multilingual JSON Dictionaries (en, ta, hi, te, kn, ml, bn, mr)
├── data/                     # Local Storage Hierarchy (pending, processing, completed, critical, archive)
├── frontend/
│   ├── index.html            # Light-Themed HTML5 Dashboard
│   ├── css/style.css         # Clean Styling System
│   └── js/app.js             # Client Interactivity & Canvas Charts
├── requirements.txt
└── README.md
```

---

## How to Run the Application

1. Open your terminal in the project directory:
   ```bash
   cd /Users/varshini/.gemini/antigravity/scratch/agri_edge_os
   ```

2. Start the Flask server:
   ```bash
   python3 backend/app.py
   ```

3. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```
