"""Configuration for 3D Boids flocking simulation."""

WINDOW = {
    "width": 1280,
    "height": 720,
    "title": "3D Boids"
}

CAMERA = {
    "fov": 90.0,
    "near_clip": 0.1,
    "far_clip": 1000.0,  # Extended draw distance
    "initial_radius": 120.0,
    "initial_theta": 45.0,
    "initial_phi": 25.0,
    "min_radius": -1500.0,
    "max_radius": 1500.0,
    "min_phi": -89.0,
    "max_phi": 89.0,
    "keyboard_rotate_speed": 60.0,
    "keyboard_zoom_speed": 20.0,
    "mouse_sensitivity": 0.3
}

GRID = {
    "base_size": 500,
    "color": (0.2, 0.2, 0.25)
}

BOIDS = {
    "count": 500000,  # Optimized with Numba + spatial grid
    "bounds": 500.0,           # Match grid size
    "max_speed": 25.0,
    "max_force": 60.0,
    "size": 1.2,
    "wall_margin": 3.0,        # Smaller margin - boids get closer to walls
    "wall_weight": 10.0,        # Stronger but shorter-range turn
    
    # Flocking behavior
    "perception_radius": 5.0,   # How far boids can see neighbors
    "separation_radius": 3.0,   # Minimum comfortable distance
    "separation_weight": 2.5,   # Avoid crowding
    "alignment_weight": 1.0,    # Match neighbor velocities
    "cohesion_weight": 1.0,     # Move toward group center
    "color_blend_rate": 1.0,    # How fast colors blend (per second)
}

COLORS = {
    "background": (0.01, 0.01, 0.02, 1.0),
    "text": (0.9, 0.9, 0.9)
}

