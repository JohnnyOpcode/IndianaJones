# 🗺️ Indiana Jones: Latent Space Vector Explorer & Novelty Discovery Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![llama.cpp](https://img.shields.io/badge/LLM-llama.cpp%20CUDA-green.svg)](https://github.com/ggerganov/llama.cpp)
[![Dashboard](https://img.shields.io/badge/UI-Interactive%20Web%20Dashboard-cyan.svg)](#-interactive-web-dashboard)

**Indiana Jones** is an autonomous, high-dimensional latent space explorer designed to traverse local LLM representations, bypass repetitive AI tropes, and discover groundbreaking, highly novel, and logically coherent conceptual "Veins of Gold."

By fusing **Persistent Cohomology & Simplicial Complex Analysis**, **PCA Subspace Vector Steering**, **Multi-Scale Kernel Density Estimation (KDE)**, **Dynamic Logit Repulsion Sampling**, and **Graph-Guided Pareto Frontier Search**, Indiana Jones transforms linear LLM generation into a guided mathematical discovery engine.

---

## ✨ Key Architectural Innovations

### 1. 📐 Persistent Cohomology & Sheaf Obstruction Engine
- **Vietoris-Rips Simplicial Complexes**: Constructs $C_0, C_1, C_2$ simplicial complexes and calculates boundary operator ranks $B_1, B_2$ over latent trajectory embeddings.
- **Betti Numbers ($b_0, b_1, b_2$) & Cohomological Dimension**: Detects topological 1-cycle loops and 2-void cavities in concept space to steer generation out of repetitive loops.
- **Sheaf Obstruction & Novelty Scoring**: Measures local-to-global coboundary section obstructions to reward structural conceptual complexity.

### 2. 📊 Multi-Scale Kernel Density Estimation (KDE) Novelty Engine
Replaces traditional single-point max similarity with Gaussian Kernel Density Estimation over past vector space embeddings:
$$D(v) = \frac{1}{K} \sum_{i=1}^K \exp\left(-\frac{\|v - e_i\|^2}{2 \sigma^2}\right) \cdot \gamma^{K - i}$$

- **Local Density vs. Global Penalty**: Detects dense semantic clusters vs. sparse voids in high-dimensional space with bounded $O(1)$ temporal windowing for scaling.
- **Lexical Entropy & Trope Penalty**: Combines vector density with n-gram surprise matrix scoring to calculate composite novelty.
- **Temporal Memory Decay ($\gamma = 0.97$)**: Exponentially decays older embeddings so the explorer can re-visit previous domains from fresh angles.

### 3. 🎯 Subspace Vector Steering & Llama-3 Instruction Chat Formatting
- **Llama 3 Chat Templates & Fallbacks**: Utilizes native `create_chat_completion` with system/user ChatML templates, backed by multi-level error recovery wrappers.
- **PCA Subspace Orthogonal Push**: Calculates principal components of recent trajectory drift and projects search seeds onto the orthogonal subspace.
- **Dynamic Sampler Logit Repulsion (`RepulsionLogitsProcessor`)**: Intercepts `llama_cpp` sampler logits to dynamically penalize overused cliché tokens.
- **16 Cross-Disciplinary Persona Lenses**: Rotates through specialized lenses (e.g. *Sheaf Cohomology*, *Synthetic Epigenetics*, *Surrealist Cybernetics*).

### 4. 🌐 Topological Graph & Pareto Frontier Traversal
- **Concept Graph Network**: Tracks semantic leaps, 2D PCA coordinates, novelty scores, and coherence values.
- **Pareto Optimal Selection**: Identifies non-dominated nodes along the multi-objective Pareto Frontier (Novelty $\times$ Coherence).
- **Smart Vector Warping**: Automatically backtracks to top Pareto nodes when hitting low-novelty plateaus.

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
