# Installation Guide

This guide covers installation for both **local development** and **Kaggle** environments.

---

## Prerequisites

- Python **3.11** or later
- `git`
- NVIDIA GPU with CUDA 12.x (recommended) — CPU-only mode is possible but impractical for training

---

## Local Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-org>/grpo-reward-hacking.git
cd grpo-reward-hacking
```

### 2. Create a virtual environment

```bash
# Using venv (standard library)
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
.venv\Scripts\activate          # Windows PowerShell
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install the project in editable mode

```bash
pip install -e .
```

This allows you to import `src.*` from anywhere in the repo without path manipulation.

### 5. Configure environment variables

Create a `.env` file in the project root (this file is gitignored):

```bash
# .env
WANDB_API_KEY=your_wandb_key_here
WANDB_ENTITY=your_wandb_entity
HF_TOKEN=your_huggingface_token   # required to download Qwen2.5
```

Load the env file (or set variables manually):

```bash
export $(cat .env | xargs)
```

### 6. Verify installation

```bash
python -c "from src.config import ExperimentConfig; print('OK')"
python -m pytest tests/ -x --no-header -q
```

---

## Flash Attention 2 (Recommended)

Flash Attention 2 significantly speeds up training. Install separately:

```bash
pip install flash-attn --no-build-isolation
```

> **Note:** FA2 requires CUDA and a compatible GPU (Ampere or newer). If unavailable,
> set `attn_implementation: "eager"` in your config.

---

## Kaggle Installation

### Option A: Setup script

In a Kaggle terminal cell:

```bash
git clone https://github.com/<your-org>/grpo-reward-hacking.git
cd grpo-reward-hacking
bash scripts/kaggle_setup.sh
```

### Option B: Notebook cells

```python
# Cell 1: Clone repo
import subprocess
subprocess.run(["git", "clone", "https://github.com/<org>/grpo-reward-hacking.git"])

# Cell 2: Install dependencies
subprocess.run(["pip", "install", "-r", "grpo-reward-hacking/requirements.txt", "-q"])

# Cell 3: Install editable
subprocess.run(["pip", "install", "-e", "grpo-reward-hacking/", "-q"])
```

### Kaggle GPU

Kaggle provides free T4 (16 GB) or P100 (16 GB) GPUs. Select one in:
**Settings → Accelerator → GPU T4 x2** (or equivalent).

Verify with:

```python
from src.utils import detect_gpu_info
print(detect_gpu_info())
```

### Kaggle W&B offline mode

If Kaggle has no internet access during the run:

```python
import os
os.environ["WANDB_MODE"] = "offline"
```

Sync later with: `wandb sync wandb/<run-dir>`

---

## Development Tools

### Code formatting

```bash
black src/ tests/ scripts/
isort src/ tests/ scripts/
```

### Linting

```bash
flake8 src/ tests/
mypy src/
```

### Tests

```bash
# Run all tests
pytest

# Run specific module
pytest tests/test_prompts.py -v

# Run with coverage
pytest --cov=src --cov-report=html
```

### Pre-commit hooks (recommended)

```bash
pip install pre-commit
pre-commit install
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ImportError: No module named 'src'` | Run `pip install -e .` from the project root |
| `CUDA out of memory` | Reduce `per_device_train_batch_size` or `num_generations` in the config |
| `ModuleNotFoundError: flash_attn` | Set `attn_implementation: eager` in config YAML |
| `wandb: ERROR` | Set `WANDB_MODE=offline` or `use_wandb: false` in config |
| `HF hub timeout` | Set `HF_DATASETS_OFFLINE=1` and pre-download datasets |
