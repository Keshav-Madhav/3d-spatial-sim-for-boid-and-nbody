# 3D Spatial Simulation - Environment Setup

## Quick Start

### Windows (PowerShell)
```powershell
.\setup_venv.ps1
```

### Linux/Mac
```bash
chmod +x setup_venv.sh
./setup_venv.sh
```

### Manual Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\Activate.ps1

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

### Core Requirements
- **pygame** - Window management and input handling
- **PyOpenGL** - 3D graphics rendering
- **numpy** - Numerical computations
- **scipy** - Scientific computing utilities
- **numba** - JIT compilation for performance

### Optional GPU Acceleration

#### NVIDIA (CUDA)
For significantly faster N-body simulations on NVIDIA GPUs:
```bash
conda install -c conda-forge cudatoolkit numba
```

#### Apple Silicon (Metal/MPS)
```bash
pip install torch
```

## Running the Simulations

### Boids Simulation
```bash
python run.py
```

### N-body Simulation
```bash
python run_nbody.py
```

### Recording & Playback
```bash
# Record simulation
python -m tools.record <scenario_name> --duration 60

# Playback recording
python -m tools.playback <scenario_name> --fps 30
```

## Performance Notes

- **CPU-only**: Uses Barnes-Hut algorithm O(n log n) for N-body
- **GPU-accelerated**: Uses brute-force O(n²) but massively parallel
  - 10-100x faster for large simulations (50K+ bodies)
  - Recommended for simulations with 10,000+ bodies

## Troubleshooting

### "No module named 'termios'" on Windows
Fixed - The code now uses cross-platform keyboard input handling.

### GPU not detected
Verify CUDA installation:
```bash
python -c "from numba import cuda; print('CUDA:', cuda.is_available())"
```

### Import errors
Make sure virtual environment is activated and all dependencies are installed:
```bash
pip install -r requirements.txt
```
