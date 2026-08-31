# 🗺️ Indiana Jones: Latent Space Vector Explorer & Novelty Discovery Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![llama.cpp](https://img.shields.io/badge/LLM-llama.cpp%20CUDA-green.svg)](https://github.com/ggerganov/llama.cpp)
[![Dashboard](https://img.shields.io/badge/UI-Interactive%20Web%20Dashboard-cyan.svg)](#-interactive-web-dashboard)

**Indiana Jones** is an autonomous, high-dimensional latent space explorer designed to traverse local LLM representations, bypass repetitive AI tropes, and discover groundbreaking, highly novel, and logically coherent conceptual "Veins of Gold."

By fusing **Persistent Cohomology & Simplicial Complex Analysis**, **PCA Subspace Vector Steering**, **Multi-Scale Kernel Density Estimation (KDE)**, **Dynamic Logit Repulsion Sampling**, and **Graph-Guided Pareto Frontier Search**, Indiana Jones transforms linear LLM generation into a guided mathematical discovery engine.

---

## ✨ Key Architectural Innovations

### 1. 📐 Persistent Homology & Cohomology Engine
- **Vietoris-Rips Simplicial Complexes ($C_0, C_1, C_2, C_3$)**: Assembles full $k$-simplices up to 3-simplices (tetrahedra) and calculates boundary operator ranks $B_1, B_2, B_3$ over latent trajectory embeddings.
- **Exact Betti Numbers ($b_0, b_1, b_2$)**: Correctly computes $b_2 = \operatorname{dim}(C_2) - \operatorname{rank}(B_2) - \operatorname{rank}(B_3)$ to eliminate false 2-void detection on dense clusters.
- **Multi-Scale Filtration & Persistence Entropy**: Evaluates topological signatures across distance percentiles and extracts harmonic 1-cycle generators via SVD nullspace decomposition.
- **Sheaf Obstruction & Novelty Scoring**: Measures local-to-global coboundary section obstructions to reward structural conceptual complexity.

### 2. 📊 Multi-Scale Cosine KDE Novelty Engine
Replaces Euclidean distance with unit-normalized Cosine Distance Kernel Density Estimation:
$$D(v) = \frac{1}{\sum w_i} \sum_{i=1}^K \exp\left(-\frac{1 - \cos(v, e_i)}{\tau}\right) \cdot \gamma^{K - i}$$

- **Calibrated High-Dimensional Bandwidth ($\tau = 0.20$)**: Prevents distance saturation in unit-hypersphere embedding spaces.
- **Lexical Entropy & Content Word Ratio**: Combines vector density with n-gram surprise matrix scoring to calculate composite novelty.
- **Dynamic Byte-Level Sampler Logit Repulsion**: Dynamic `RepulsionLogitsProcessor` intercepts `llama_cpp` sampler logits at the byte level to actively penalize clichés and buzzwords.

### 3. 🎯 Subspace Vector Steering & Domain Anchor Preservation
- **Domain Anchor Grounding**: Preserves the primary starting concept in all prompt derivations to prevent unanchored semantic drift while exploring orthogonal frontiers.
- **PCA Subspace Orthogonal Push & Momentum Steering**: Projects candidate seeds onto orthogonal subspaces and injects momentum towards discovered gold veins.
- **Semantic Contrast Theme Synthesis**: Extracts high-dimensional orthogonal divergence themes and feeds them as explicit steering cues to the generative LLM.
- **16 Balanced Multidisciplinary Persona Lenses**: Rotates across a wide spectrum of scientific and philosophical analytical frameworks.

### 4. 🌐 Topological Graph & Pareto Frontier Traversal
- **Continuous Multi-Objective Evaluation**: Scores both novelty and continuous semantic coherence on all exploration steps.
- **Pareto Optimal Selection**: Identifies non-dominated nodes along the Pareto Frontier (Novelty $\times$ Coherence) to guide backtracking and exploration jumps.
- **Global PCA Coordinate Alignment**: Re-projects historical embeddings dynamically for consistent 2D scatter visualization.

### 5. 🖥️ Interactive Live Expedition Web Dashboard
- **Real-Time Telemetry**: Lightweight HTTP server hosting live state at `http://localhost:8000`.
- **2D PCA Latent Scatter Map**: Visualizes the high-dimensional latent path with downsampled payload optimization.
- **Real-Time Telemetry Charts**: Canvas-based line graphs for Novelty, Coherence, and Cohomological Dimension.
- **Interactive Control Panel**: Live controls for Vector Warps and threshold sliders.

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

### Command Line Interface

Launch the explorer using flexible command-line options:

```bash
# Default run (100 steps with dashboard at http://localhost:8000)
python indiana_jones.py

# Custom concept and steps
python indiana_jones.py --concept "Synthetic biology and topological memory" --steps 50

# Custom GGUF model path and output directory
python indiana_jones.py -m ./LLM/MyModel.gguf -o ./results -g 0.85
```

### CLI Arguments Summary

| Flag | Short | Default | Description |
| --- | --- | --- | --- |
| `--model-path` | `-m` | `./LLM/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf` | Path to GGUF model file |
| `--concept` | `-c` | `"The relationship between truth and confabulations"` | Starting seed concept |
| `--steps` | `-s` | `100` | Total expedition steps |
| `--gold-threshold` | `-g` | `0.90` | Novelty threshold for gold discovery |
| `--port` | `-p` | `8000` | Web dashboard server port |
| `--no-dashboard` | | `False` | Run without starting web dashboard |
| `--output-dir` | `-o` | `.` | Output directory for JSON journals |
| `--temperature` | `-t` | `0.90` | Initial LLM sampling temperature |

---

## 📂 Project Structure

```
IndianaJones/
├── engines/
│   ├── cohomology.py       # Persistent Cohomology & Sheaf Obstruction Engine
│   ├── novelty.py          # Multi-Scale KDE & Repulsion Logits Processor
│   ├── steering.py         # Subspace Orthogonal Steering & Topological Push
│   └── graph.py            # Latent Graph & Pareto Frontier Network
├── core/
│   ├── config.py           # Configuration Dataclass & CLI Argument Parser
│   ├── dll_setup.py        # Automatic Windows CUDA/DLL loader setup
│   └── explorer.py         # LatentExplorer Core Orchestrator & LLM Handler
├── indiana_jones.py        # Primary CLI Entry Point
├── dashboard_server.py     # HTTP/API server for telemetry & live controls
├── dashboard.html          # Interactive Web Dashboard UI (Canvas/Chart JS)
├── requirements.txt        # Python dependency manifest
└── README.md               # Project documentation
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
