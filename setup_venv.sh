#!/bin/bash
# Bash script to set up virtual environment for the 3D simulation project (Linux/Mac)

echo "Creating virtual environment..."
python3 -m venv .venv

echo -e "\nActivating virtual environment..."
source .venv/bin/activate

echo -e "\nUpgrading pip..."
pip install --upgrade pip

echo -e "\nInstalling requirements..."
pip install -r requirements.txt

echo -e "\n========================================"
echo "Virtual environment setup complete!"
echo "========================================"
echo -e "\nTo activate the environment in the future, run:"
echo "  source .venv/bin/activate"
echo -e "\nTo deactivate, run:"
echo "  deactivate"
echo -e "\nFor GPU acceleration:"
echo "  Apple Silicon: pip install torch"
echo "  NVIDIA: conda install -c conda-forge cudatoolkit numba"
echo -e "\nTo run the simulations:"
echo "  python run.py          # Boids simulation"
echo "  python run_nbody.py    # N-body simulation"
