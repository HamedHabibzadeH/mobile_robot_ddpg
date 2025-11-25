This repository contains the full code for a mobile robot navigation environment with LiDAR and training using DDPG.

## Quick Installation

Python 3.9–3.11 is recommended.

```bash
# (Optional) create a virtual environment
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

# Upgrade packaging tools
python -m pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt

# Install PyTorch (depending on your system)
# CPU only:
pip install torch --index-url https://download.pytorch.org/whl/cpu
# or follow the CUDA-specific instructions: https://pytorch.org/get-started/locally/
