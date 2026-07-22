# 🗺️ Indiana Jones: Latent Space Vector Explorer & Novelty Discovery Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![llama.cpp](https://img.shields.io/badge/LLM-llama.cpp%20CUDA-green.svg)](https://github.com/ggerganov/llama.cpp)
[![Dashboard](https://img.shields.io/badge/UI-Interactive%20Web%20Dashboard-cyan.svg)](#-interactive-web-dashboard)

**Indiana Jones** is an autonomous, high-dimensional latent space explorer designed to traverse local LLM representations, bypass repetitive AI tropes, and discover groundbreaking, highly novel, and logically coherent conceptual "Veins of Gold."

By fusing **PCA Subspace Vector Steering**, **Multi-Scale Kernel Density Estimation (KDE)**, **Dynamic Logit Repulsion Sampling**, and **Graph-Guided Pareto Frontier Search**, Indiana Jones transforms linear LLM generation into a guided mathematical discovery engine.

---

## ✨ Key Architectural Innovations

### 1. 📊 Multi-Scale Kernel Density Estimation (KDE) Novelty Engine

Replaces traditional single-point max similarity with Gaussian Kernel Density Estimation over past vector space embeddings:
$$D(v) = \frac{1}{K} \sum_{i=1}^K \exp\left(-\frac{\|v - e_i\|^2}{2 \sigma^2}\right) \cdot \gamma^{K - i}$$

- **Local Density vs. Global Penalty**: Detects dense semantic clusters (over-explored regions) vs. sparse voids in high-dimensional space.
- **Lexical Entropy & Trope Penalty**: Combines dense vector density with n-gram surprise matrix scoring to calculate composite novelty score $N = 0.7 \cdot N_{\text{KDE}} + 0.3 \cdot N_{\text{Surprise}}$.
- **Temporal Memory Decay ($\gamma = 0.97$)**: Exponentially decays older embeddings so the explorer can re-visit previous domains from fresh angles without global penalty.

### 2. 🎯 Subspace Vector Steering & Dynamic Logit Repulsion

- **PCA Subspace Orthogonal Push**: Calculates principal components of recent trajectory drift and projects search seeds onto the orthogonal subspace to force exploration into unmapped latent dimensions.
- **Dynamic Sampler Logit Repulsion (`RepulsionLogitsProcessor`)**: Intercepts `llama_cpp` sampler logits to dynamically penalize overused cliché tokens (_"quantum foam", "tapestry", "forgotten cathedral", "sentient", "whisper", "fabric of reality"_) in real-time.
- **Cross-Disciplinary Persona Lenses**: Rotates through 12 specialized lenses (e.g. _Synthetic Epigenetics_, _Topological Fluid Dynamics_, _Surrealist Cybernetics_, _Crystalline Semiotics_).

### 3. 🌐 Topological Graph & Pareto Frontier Traversal

- **Concept Graph Network**: Tracks semantic leaps, PCA 2D coordinates, novelty scores, and coherence values.
- **Pareto Optimal Selection**: Identifies non-dominated nodes along the multi-objective Pareto Frontier (Novelty $\times$ Coherence $\times$ Domain Distance).
- **Smart Graph Backtracking**: Automatically backtracks to top Pareto nodes when hitting low-novelty plateaus instead of wandering linearly.

### 4. 🖥️ Interactive Live Expedition Web Dashboard

- **Real-Time Telemetry**: Automatically hosts a lightweight web server at `http://localhost:8000`.
- **2D PCA Latent Scatter Map**: Visualizes the high-dimensional latent path, active vector trajectories, and highlighted Gold Vein discoveries.
- **Real-Time Telemetry Charts**: Canvas-based line graphs for Novelty, Coherence, Explorer Energy, and Active Temperature.
- **Interactive Control Panel**: Live sliders for Orthogonal Push weight, Logit Repulsion penalty, Gold threshold, and a **Force Quantum Vector Warp** button.

---

## 🏛️ System Architecture

```
                                  ┌───────────────────────────┐
                                  │   Starting Concept Seed   │
                                  └─────────────┬─────────────┘
                                                │
                                                ▼
┌───────────────────────────┐     ┌───────────────────────────┐
│  RepulsionLogitsProcessor │ ──► │  llama_cpp Local LLM      │
│  (Cliché Trope Penalty)   │     │ (Meta-Llama-3-8B-Instruct)│
└───────────────────────────┘     └─────────────┬─────────────┘
                                                │
                                                ▼
                                  ┌───────────────────────────┐
                                  │  Sentence Transformers    │
                                  │  ('all-MiniLM-L6-v2')     │
                                  └─────────────┬─────────────┘
                                                │
                                                ▼
┌───────────────────────────┐     ┌───────────────────────────┐
│ Topological Concept Graph │ ◄── │    NoveltyEngine (KDE)    │
│  & Pareto Frontier Search │     │ + Lexical Entropy Matrix  │
└─────────────┬─────────────┘     └─────────────┬─────────────┘
              │                                 │
              ▼                                 ▼
┌───────────────────────────┐     ┌───────────────────────────┐
│  VectorSteering (PCA Push │     │   Live Web Dashboard UI   │
│   & Gold Momentum)        │     │  (http://localhost:8000)  │
└───────────────────────────┘     └───────────────────────────┘
```

---

## ⚡ Quickstart & Installation

### Prerequisites

- Python 3.10 or higher
- NVIDIA GPU with CUDA support (Recommended for GGUF acceleration)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/IndianaJones.git
cd IndianaJones
```

### 2. Create Virtual Environment & Install Dependencies

```bash
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Download LLM Model Weights

Download Meta Llama 3 8B Instruct GGUF (e.g. `Meta-Llama-3-8B-Instruct.Q4_K_M.gguf`) into the `LLM/` directory:

```bash
mkdir LLM
# Place your GGUF file inside ./LLM/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
```

---

## 🚀 Usage

### Start an Expedition

Launch the explorer with default parameters:

```bash
python indiana_jones.py
```

### Open the Interactive Dashboard

Open your web browser and navigate to:

```
http://localhost:8000
```

From the dashboard you can:

- Observe the **2D PCA Vector Scatter Plot** updating live with every step.
- Track real-time **KDE Novelty vs. Coherence** telemetry charts.
- View discovered **Gold Vein** entries.
- Adjust **Logit Repulsion Strength**, **Orthogonal Push Weight**, or trigger an instant **Quantum Vector Warp**.

---

## 📂 Project Structure

```
IndianaJones/
├── indiana_jones.py        # Core Explorer, NoveltyEngine & VectorSteering
├── dashboard_server.py     # HTTP/API server for telemetry & live controls
├── dashboard.html          # Interactive Web Dashboard UI (Canvas/Chart JS)
├── requirements.txt        # Python dependency manifest
├── .gitignore              # Git ignore configuration
├── LICENSE                 # MIT License
└── README.md               # Project documentation
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
