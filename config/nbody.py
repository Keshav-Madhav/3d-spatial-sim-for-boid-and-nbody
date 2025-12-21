"""Configuration for N-body gravitational simulation."""

# =============================================================================
# PERFORMANCE PRESETS - Choose one by uncommenting
# =============================================================================

# PRESET: ULTRA (1M bodies, ~1 FPS) - for screenshots/recordings
# BODY_COUNT = 1_000_000
# THETA = 0.9  # Very approximate but fast

# PRESET: HIGH (500K bodies, ~2-3 FPS) - impressive visuals
# BODY_COUNT = 500_000
# THETA = 0.85

# PRESET: MEDIUM (250K bodies, ~5-10 FPS) - good balance
BODY_COUNT = 150_000
THETA = 0.8

# PRESET: SMOOTH (100K bodies, ~15-30 FPS) - interactive
# BODY_COUNT = 100_000
# THETA = 0.75

# PRESET: FAST (50K bodies, ~30-60 FPS) - very smooth
# BODY_COUNT = 50_000
# THETA = 0.7

# =============================================================================

WINDOW = {
    "width": 1280,
    "height": 720,
    "title": "N-Body Gravitational Simulation"
}

CAMERA = {
    "fov": 75.0,
    "near_clip": 0.1,
    "far_clip": 5000.0,  # Extended for large simulations
    "initial_radius": 800.0,
    "initial_theta": 45.0,
    "initial_phi": 35.0,
    "min_radius": -3000.0,
    "max_radius": 3000.0,
    "min_phi": -89.0,
    "max_phi": 89.0,
    "keyboard_rotate_speed": 60.0,
    "keyboard_zoom_speed": 100.0,
    "mouse_sensitivity": 0.3
}

GRID = {
    "base_size": 1000,
    "color": (0.08, 0.08, 0.12)
}

# N-body simulation parameters
NBODY = {
    "count": BODY_COUNT,           # Number of bodies
    "spawn_radius": 500.0,         # Initial spawn radius (bodies can escape freely)
    
    # Physics parameters
    "G": 0.1,                      # Gravitational constant (tuned for visualization)
    "theta": THETA,                # Barnes-Hut opening angle (0.5-1.0, higher = faster but less accurate)
    "softening": 2.0,              # Softening length to prevent singularities
    "damping": 1.0,                # No damping - pure Newtonian physics
    
    # Initial distribution: "galaxy", "spiral", "sphere", "collision", "uniform"
    "distribution": "galaxy",
    
    # Rendering
    "point_size": 1.5,             # Size of rendered points
    "max_speed_color": 15.0,       # Velocity for max brightness
}

COLORS = {
    "background": (0.0, 0.0, 0.02, 1.0),  # Deep space black
    "text": (0.7, 0.8, 0.9)
}

