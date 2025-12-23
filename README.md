# 3D N-Body Simulation

A high-performance gravitational N-body simulation using Barnes-Hut algorithm with octree spatial partitioning. Supports up to 50 million bodies with offline rendering for smooth playback.

## Features

- **Barnes-Hut Algorithm**: O(N log N) gravitational force calculation
- **Numba JIT**: Parallel CPU acceleration with SIMD
- **GPU Support**: CUDA (NVIDIA) and Metal (Apple Silicon) backends
- **Offline Rendering**: Pre-compute frames for smooth playback
- **Multiple Distributions**: Galaxy, collision, spiral, cluster, ring, and more
- **Video Export**: 4K 60fps with multiple codecs

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Optional: GPU Support

```bash
# NVIDIA (CUDA)
conda install cudatoolkit numba

# Apple Silicon (Metal) - already included via PyTorch MPS
pip install torch
```

---

## Recording Simulations

### Interactive Menu
```bash
python -m tools.record
```
Shows preset selection menu with optional overrides for bodies, frames, and theta.

### Using Presets

```bash
# By preset name
python -m tools.record --preset quick_galaxy
python -m tools.record --preset galaxy_epic
python -m tools.record --preset 4k_collision_1m

# By preset index (from menu)
python -m tools.record --preset-id 15
```

### Override Parameters

```bash
# Override bodies (supports k/m suffix)
python -m tools.record --preset quick_galaxy --bodies 500k
python -m tools.record --preset quick_galaxy -n 1m

# Override frames
python -m tools.record --preset quick_galaxy --frames 2000
python -m tools.record --preset quick_galaxy -f 2000

# Override theta (accuracy: lower = more accurate, slower)
python -m tools.record --preset quick_galaxy --theta 0.5
python -m tools.record --preset quick_galaxy -t 0.5

# Override time step
python -m tools.record --preset quick_galaxy --dt 0.1

# Combine multiple overrides
python -m tools.record --preset quick_galaxy -n 500k -f 3000 -t 0.6
```

### Managing Recordings

```bash
# List all recordings with progress
python -m tools.record --list

# Check status of specific recording
python -m tools.record --status bar_galaxy

# Resume interrupted recording
python -m tools.record --resume
python -m tools.record --resume bar_galaxy

# Extend existing recording by N frames
python -m tools.record --extend 3000 bar_galaxy
```

---

## Playback

```bash
# Basic playback
python -m tools.playback <session_name>

# Custom FPS
python -m tools.playback bar_galaxy --fps 60

# Loop playback
python -m tools.playback bar_galaxy --loop

# Start from specific frame
python -m tools.playback bar_galaxy --start 500
```

### Playback Controls
- `WASD` - Move camera
- `Mouse` - Look around
- `Scroll` - Zoom
- `Space` - Pause/Resume
- `R` - Reset camera
- `Q/ESC` - Quit

---

## Video Export

### Interactive Mode (Recommended)
```bash
# Interactive export with menus
python -m tools.export bar_galaxy

# Or explicitly request interactive
python -m tools.export bar_galaxy -i
```

### Command Line Options
```bash
# List exportable recordings
python -m tools.export --list

# Basic export
python -m tools.export bar_galaxy --fps 30 --resolution 1080p

# High quality 4K export
python -m tools.export bar_galaxy --resolution 4k --fps 60 --quality high

# Codec options
python -m tools.export bar_galaxy --codec h264    # Most compatible
python -m tools.export bar_galaxy --codec h265    # Better compression
python -m tools.export bar_galaxy --codec vp9     # Open format

# Camera modes
python -m tools.export bar_galaxy --camera orbit      # Horizontal orbit (default)
python -m tools.export bar_galaxy --camera fixed      # Static position
python -m tools.export bar_galaxy --camera zoomout    # Constant zoom out
python -m tools.export bar_galaxy --camera zoomin     # Constant zoom in
python -m tools.export bar_galaxy --camera spiral     # Spiral motion
python -m tools.export bar_galaxy --camera cinematic  # Dramatic sweep
python -m tools.export bar_galaxy --camera flyby      # Flyby effect
python -m tools.export bar_galaxy --camera topdown    # Top-down view

# Full example
python -m tools.export bar_galaxy \
    --resolution 4k \
    --fps 60 \
    --quality ultra \
    --codec h265 \
    --camera cinematic
```

---

## Live Simulation

Run real-time interactive simulation (limited body count for smooth FPS):

```bash
# Default (100K bodies)
python run_nbody.py

# Custom body count
python run_nbody.py --bodies 50000

# Adjust accuracy
python run_nbody.py --bodies 100000 --theta 0.8
```

### Live Controls
- `WASD` - Move camera
- `Mouse` - Look around
- `Scroll` - Zoom in/out
- `Space` - Pause simulation
- `R` - Reset simulation
- `G` - Toggle grid
- `Q/ESC` - Quit

---

## Preset Categories

| Category | Description | Est. Time |
|----------|-------------|-----------|
| **TINY** | Testing (10-20K bodies) | 3-5 seconds |
| **FAST** | Quick renders (50-100K bodies) | 10-25 seconds |
| **CINEMATIC** | Production quality (300-500K) | 30-60 minutes |
| **CINEMATIC_4K** | 4K 60fps, high accuracy | 3-18 hours |
| **ARTISTIC** | Visually striking | 5-10 minutes |
| **SCIENTIFIC** | Physically accurate | 10-50 minutes |
| **CHAOS** | Wild simulations | 3-15 minutes |
| **MEGA** | 1M bodies | 40-60 minutes |
| **EXTREME** | 5-50M bodies | 17 min - 2 hours |

---

## Distributions

| Distribution | Description |
|--------------|-------------|
| `galaxy` | Classic spiral disk galaxy |
| `collision` | Two galaxies colliding |
| `spiral` | Multi-arm spiral galaxy |
| `cluster` | Dense globular cluster (Plummer model) |
| `ring` | Saturn-like ring structure |
| `shell` | Hollow spherical shell |
| `binary` | Binary star system with disks |
| `elliptical` | Elliptical galaxy (3D bulge) |
| `bar` | Barred spiral galaxy |
| `stream` | Tidal stream / stellar river |
| `filament` | Cosmic web filament |
| `explosion` | Expanding supernova shell |
| `vortex` | Swirling vortex structure |
| `pleiades` | Star cluster with nebulosity |
| `double_helix` | DNA-like double helix structure |
| `accretion_disk` | Black hole accretion disk with jets |
| `torus` | Donut-shaped torus |
| `hourglass` | Binary star hourglass nebula |
| `fibonacci` | Fibonacci spiral pattern |
| `triple` | Three galaxies in triangle formation |
| `rosette` | Flower-like orbital rosette |
| `dyson` | Dyson sphere megastructure |

---

## Technical Details

### Barnes-Hut Algorithm
- **θ (theta)**: Controls accuracy/speed tradeoff
  - `0.3-0.5`: High accuracy (slow)
  - `0.7-0.9`: Balanced
  - `1.0-1.5`: Fast (approximate)

### Substeps
- Multiple physics steps per frame for smoother motion
- Higher substeps = more accurate but slower

### Softening
- Prevents singularities when particles get close
- Higher values = smoother but less realistic close encounters

---

## File Structure

```
recordings/
├── bar_galaxy/
│   ├── metadata.json      # Recording settings
│   ├── frame_0000.npz     # Position + color data
│   ├── frame_0001.npz
│   ├── ...
│   └── state_1950.npz     # Resume checkpoint
└── bar_galaxy.mp4         # Exported video
```

---

## Performance Tips

1. **Start small**: Test with TINY/FAST presets first
2. **Adjust theta**: Higher theta = faster but less accurate
3. **Use offline rendering**: For >200K bodies, record offline
4. **GPU acceleration**: Automatically used when available
5. **Extend recordings**: Use `--extend` to add frames without recomputing

