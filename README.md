# Advanced Trajectory Flow Matching (TFM)

A simulation-free generative framework for **pedestrian trajectory prediction**, built on **Trajectory Flow Matching (TFM)** with **GRU** and **Neural CDE** extensions.  
This project models **multimodal futures**, **uncertainty**, and **irregularly sampled trajectories** in a continuous-time setting.

---

## 🚀 Overview

Predicting human trajectories is challenging due to stochastic behavior, irregular sampling, and long-range social interactions.  
Traditional discrete-time models (RNNs, GNNs, Transformers) struggle to capture continuous dynamics and uncertainty.

**Trajectory Flow Matching (TFM)** offers a *simulation-free* alternative to Neural SDE training by aligning data and model flows directly—avoiding costly solver backpropagation.  

We extend TFM with:
- **Sequential encoders (GRUs)** for long-term dependencies.
- **Continuous-time encoders (Neural CDEs)** for irregular sampling.
- **Uncertainty modeling** via data-dependent diffusion.

---
## 🧭 Environment Setup (with `uv`)

This project uses **[uv](https://github.com/astral-sh/uv)** — a fast, modern Python package and environment manager.  
Follow the steps below to reproduce the exact development environment used for this project.

### 1️⃣ Clone the repository

```bash
git clone https://github.com/<your-org-or-username>/TrajectoryFlowMatching.git
cd TrajectoryFlowMatching
```

### 2️⃣ Create and activate a virtual environment

```bash
uv venv --python 3.10 --prompt trajFM
source .venv/bin/activate
```
### 3️⃣ Sync dependencies
All dependencies (including PyTorch from the CUDA 11.3 wheel index) are stored in the project metadata.
To install everything exactly as in the original setup, run:

```bash
uv sync
```

## Alternative Setup (without uv)
If `uv` is not available, you can reproduce the environment with classic `pip`:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 \
  --index-url https://download.pytorch.org/whl/cu113
```

## 📊 Datasets

- **ETH/UCY** pedestrian trajectory benchmark  
  - Leave-one-scene-out protocol  
  - 8 observed steps → 12 predicted steps (3.2s → 4.8s)  
  - Preprocessed trajectories normalized to local scene coordinates

---

## ⚙️ Training

