"""
Recording Presets Library
=========================

A collection of pre-configured N-body simulation presets organized by category.
Each preset includes optimal physics parameters, spawn distribution, and render settings.

Categories:
- CINEMATIC: Beautiful, production-quality renders (slower, more accurate)
- FAST: Quick renders for testing and fun
- SCIENTIFIC: Physically accurate simulations  
- CHAOS: Wild, unpredictable simulations
- ARTISTIC: Visually striking configurations
"""

import numpy as np
from typing import Dict, List, Tuple

# =============================================================================
# SPAWN DISTRIBUTIONS
# =============================================================================

DISTRIBUTIONS = {
    "galaxy": "Classic spiral disk galaxy",
    "collision": "Two galaxies colliding",
    "spiral": "Multi-arm spiral galaxy",
    "sphere": "Uniform spherical distribution",
    "ring": "Saturn-like ring structure",
    "shell": "Hollow spherical shell",
    "cluster": "Dense star cluster (globular)",
    "binary": "Binary star system with disks",
    "elliptical": "Elliptical galaxy (3D bulge)",
    "bar": "Barred spiral galaxy",
    "stream": "Tidal stream / stellar river",
    "filament": "Cosmic web filament",
    "explosion": "Expanding supernova shell",
    "vortex": "Swirling vortex structure",
    "cube": "Cubic lattice (for testing)",
    "pleiades": "Star cluster with nebulosity",
}


def generate_distribution(distribution: str, n: int, R: float, G: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate initial conditions for various distributions.
    
    Returns:
        positions: (n, 3) array
        velocities: (n, 3) array
        masses: (n,) array
    """
    positions = np.zeros((n, 3), dtype=np.float64)
    velocities = np.zeros((n, 3), dtype=np.float64)
    masses = np.ones(n, dtype=np.float64)
    
    if distribution == "galaxy":
        # Exponential disk galaxy
        r = np.random.exponential(R * 0.3, n)
        theta = np.random.uniform(0, 2 * np.pi, n)
        z = np.random.normal(0, R * 0.02, n)
        
        positions[:, 0] = r * np.cos(theta)
        positions[:, 1] = z
        positions[:, 2] = r * np.sin(theta)
        
        orbital_speed = np.sqrt(G * n * 0.001 / (r + 1.0))
        velocities[:, 0] = -orbital_speed * np.sin(theta)
        velocities[:, 2] = orbital_speed * np.cos(theta)
        velocities[:, 1] = np.random.normal(0, orbital_speed * 0.1, n)
        
    elif distribution == "collision":
        # Two galaxies on collision course
        half = n // 2
        
        # Galaxy 1 (centered at -R, moving right)
        r1 = np.random.exponential(R * 0.25, half)
        theta1 = np.random.uniform(0, 2 * np.pi, half)
        positions[:half, 0] = r1 * np.cos(theta1) - R * 0.8
        positions[:half, 1] = np.random.normal(0, R * 0.02, half)
        positions[:half, 2] = r1 * np.sin(theta1)
        
        orbital_speed1 = np.sqrt(G * half * 0.001 / (r1 + 1.0))
        velocities[:half, 0] = -orbital_speed1 * np.sin(theta1) + 3.0
        velocities[:half, 2] = orbital_speed1 * np.cos(theta1)
        
        # Galaxy 2 (centered at +R, moving left)
        r2 = np.random.exponential(R * 0.25, n - half)
        theta2 = np.random.uniform(0, 2 * np.pi, n - half)
        positions[half:, 0] = r2 * np.cos(theta2) + R * 0.8
        positions[half:, 1] = np.random.normal(0, R * 0.02, n - half) + R * 0.3
        positions[half:, 2] = r2 * np.sin(theta2)
        
        orbital_speed2 = np.sqrt(G * (n - half) * 0.001 / (r2 + 1.0))
        velocities[half:, 0] = -orbital_speed2 * np.sin(theta2) - 2.0
        velocities[half:, 2] = orbital_speed2 * np.cos(theta2)
        
    elif distribution == "spiral":
        # Multi-arm spiral galaxy
        disk_r = np.random.exponential(R * 0.25, n)
        disk_r = np.clip(disk_r, R * 0.02, R * 0.9)
        
        spiral_tightness = 0.3
        base_theta = np.log(disk_r / (R * 0.05) + 1) / spiral_tightness
        arm_assignment = np.random.randint(0, 4, n)
        arm_offset = arm_assignment * (2 * np.pi / 4)
        theta = base_theta + arm_offset + np.random.normal(0, 0.2, n)
        
        positions[:, 0] = disk_r * np.cos(theta)
        positions[:, 2] = disk_r * np.sin(theta)
        positions[:, 1] = np.random.normal(0, R * 0.01, n) * (1 - disk_r / R)
        
        orbital_speed = np.sqrt(G * n * 0.001 / (disk_r + 1.0))
        velocities[:, 0] = -orbital_speed * np.sin(theta)
        velocities[:, 2] = orbital_speed * np.cos(theta)
        velocities += np.random.normal(0, orbital_speed[:, np.newaxis] * 0.05, (n, 3))
        
    elif distribution == "ring":
        # Saturn-like ring with central mass concentration
        core_n = n // 10
        ring_n = n - core_n
        
        # Dense core
        r_core = np.random.exponential(R * 0.05, core_n)
        phi_core = np.random.uniform(0, 2 * np.pi, core_n)
        cos_theta_core = np.random.uniform(-1, 1, core_n)
        sin_theta_core = np.sqrt(1 - cos_theta_core**2)
        
        positions[:core_n, 0] = r_core * sin_theta_core * np.cos(phi_core)
        positions[:core_n, 1] = r_core * cos_theta_core
        positions[:core_n, 2] = r_core * sin_theta_core * np.sin(phi_core)
        masses[:core_n] = 10.0  # Heavy core particles
        
        # Ring
        ring_r = np.random.uniform(R * 0.4, R * 0.8, ring_n)
        ring_theta = np.random.uniform(0, 2 * np.pi, ring_n)
        ring_z = np.random.normal(0, R * 0.01, ring_n)
        
        positions[core_n:, 0] = ring_r * np.cos(ring_theta)
        positions[core_n:, 1] = ring_z
        positions[core_n:, 2] = ring_r * np.sin(ring_theta)
        
        orbital_speed = np.sqrt(G * core_n * 10 * 0.001 / ring_r)
        velocities[core_n:, 0] = -orbital_speed * np.sin(ring_theta)
        velocities[core_n:, 2] = orbital_speed * np.cos(ring_theta)
        
    elif distribution == "shell":
        # Hollow spherical shell (like a supernova remnant)
        r_inner = R * 0.7
        r_outer = R * 0.9
        
        u = np.random.uniform(0, 1, n)
        r = (r_inner**3 + u * (r_outer**3 - r_inner**3)) ** (1/3)
        
        phi = np.random.uniform(0, 2 * np.pi, n)
        cos_theta = np.random.uniform(-1, 1, n)
        sin_theta = np.sqrt(1 - cos_theta**2)
        
        positions[:, 0] = r * sin_theta * np.cos(phi)
        positions[:, 1] = r * cos_theta
        positions[:, 2] = r * sin_theta * np.sin(phi)
        
        # Slight expansion
        velocities[:, 0] = positions[:, 0] * 0.01
        velocities[:, 1] = positions[:, 1] * 0.01
        velocities[:, 2] = positions[:, 2] * 0.01
        
    elif distribution == "cluster":
        # Globular cluster (Plummer model)
        a = R * 0.3  # Plummer scale radius
        
        # Plummer distribution
        u = np.random.uniform(0, 1, n)
        r = a / np.sqrt(u**(-2/3) - 1)
        r = np.clip(r, 0, R * 2)
        
        phi = np.random.uniform(0, 2 * np.pi, n)
        cos_theta = np.random.uniform(-1, 1, n)
        sin_theta = np.sqrt(1 - cos_theta**2)
        
        positions[:, 0] = r * sin_theta * np.cos(phi)
        positions[:, 1] = r * cos_theta
        positions[:, 2] = r * sin_theta * np.sin(phi)
        
        # Isotropic velocities (virial equilibrium approximation)
        sigma = np.sqrt(G * n * 0.001 / (6 * a))
        velocities = np.random.normal(0, sigma, (n, 3))
        
    elif distribution == "binary":
        # Binary star system with two disks
        n1 = n // 2
        n2 = n - n1
        separation = R * 0.6
        
        # Star 1 disk
        r1 = np.random.exponential(R * 0.15, n1)
        theta1 = np.random.uniform(0, 2 * np.pi, n1)
        positions[:n1, 0] = r1 * np.cos(theta1) - separation / 2
        positions[:n1, 1] = np.random.normal(0, R * 0.01, n1)
        positions[:n1, 2] = r1 * np.sin(theta1)
        
        orbital_speed1 = np.sqrt(G * n1 * 0.001 / (r1 + 1.0))
        binary_orbital = np.sqrt(G * n * 0.0005 / separation)
        velocities[:n1, 0] = -orbital_speed1 * np.sin(theta1)
        velocities[:n1, 2] = orbital_speed1 * np.cos(theta1) - binary_orbital
        
        # Star 2 disk (tilted)
        r2 = np.random.exponential(R * 0.15, n2)
        theta2 = np.random.uniform(0, 2 * np.pi, n2)
        tilt = np.pi / 6  # 30 degree tilt
        
        x2 = r2 * np.cos(theta2) + separation / 2
        y2 = r2 * np.sin(theta2) * np.sin(tilt)
        z2 = r2 * np.sin(theta2) * np.cos(tilt)
        
        positions[n1:, 0] = x2
        positions[n1:, 1] = y2
        positions[n1:, 2] = z2
        
        orbital_speed2 = np.sqrt(G * n2 * 0.001 / (r2 + 1.0))
        velocities[n1:, 0] = -orbital_speed2 * np.sin(theta2)
        velocities[n1:, 2] = orbital_speed2 * np.cos(theta2) + binary_orbital
        
    elif distribution == "elliptical":
        # Elliptical galaxy (de Vaucouleurs profile approximation)
        # 3D triaxial ellipsoid
        a, b, c = R * 0.5, R * 0.4, R * 0.3  # Semi-axes
        
        r = np.random.exponential(R * 0.2, n)
        phi = np.random.uniform(0, 2 * np.pi, n)
        cos_theta = np.random.uniform(-1, 1, n)
        sin_theta = np.sqrt(1 - cos_theta**2)
        
        positions[:, 0] = a * r / R * sin_theta * np.cos(phi)
        positions[:, 1] = b * r / R * cos_theta
        positions[:, 2] = c * r / R * sin_theta * np.sin(phi)
        
        # Random velocities (pressure-supported)
        sigma = np.sqrt(G * n * 0.0005 / R)
        velocities = np.random.normal(0, sigma, (n, 3))
        
    elif distribution == "bar":
        # Barred spiral galaxy
        bar_n = n // 3
        disk_n = n - bar_n
        
        # Central bar
        bar_length = R * 0.4
        bar_r = np.random.exponential(bar_length * 0.3, bar_n)
        bar_theta = np.random.uniform(-np.pi/6, np.pi/6, bar_n)  # Narrow angle
        
        positions[:bar_n, 0] = bar_r * np.cos(bar_theta)
        positions[:bar_n, 1] = np.random.normal(0, R * 0.02, bar_n)
        positions[:bar_n, 2] = bar_r * np.sin(bar_theta) * 0.3  # Thin bar
        
        bar_speed = np.sqrt(G * n * 0.0005 / (bar_r + 1))
        velocities[:bar_n, 0] = -bar_speed * np.sin(bar_theta)
        velocities[:bar_n, 2] = bar_speed * np.cos(bar_theta)
        
        # Outer spiral disk
        disk_r = np.random.uniform(R * 0.3, R * 0.8, disk_n)
        spiral_theta = np.log(disk_r / (R * 0.1) + 1) / 0.4
        arm = np.random.randint(0, 2, disk_n)
        disk_theta = spiral_theta + arm * np.pi + np.random.normal(0, 0.3, disk_n)
        
        positions[bar_n:, 0] = disk_r * np.cos(disk_theta)
        positions[bar_n:, 1] = np.random.normal(0, R * 0.01, disk_n)
        positions[bar_n:, 2] = disk_r * np.sin(disk_theta)
        
        disk_speed = np.sqrt(G * n * 0.001 / (disk_r + 1))
        velocities[bar_n:, 0] = -disk_speed * np.sin(disk_theta)
        velocities[bar_n:, 2] = disk_speed * np.cos(disk_theta)
        
    elif distribution == "stream":
        # Tidal stream / stellar river
        t = np.random.uniform(0, 1, n)
        
        # Sinusoidal path through space
        length = R * 3
        positions[:, 0] = (t - 0.5) * length
        positions[:, 1] = np.sin(t * 4 * np.pi) * R * 0.3 + np.random.normal(0, R * 0.03, n)
        positions[:, 2] = np.cos(t * 4 * np.pi) * R * 0.3 + np.random.normal(0, R * 0.03, n)
        
        # Velocity along stream with some dispersion
        velocities[:, 0] = 5.0 + np.random.normal(0, 0.5, n)
        velocities[:, 1] = np.random.normal(0, 0.3, n)
        velocities[:, 2] = np.random.normal(0, 0.3, n)
        
    elif distribution == "filament":
        # Cosmic web filament with nodes
        num_nodes = 5
        node_positions = np.linspace(-R, R, num_nodes)
        
        # Assign particles to nodes with varying density
        node_idx = np.random.choice(num_nodes, n, p=[0.3, 0.15, 0.1, 0.15, 0.3])
        
        for i in range(num_nodes):
            mask = node_idx == i
            node_n = np.sum(mask)
            if node_n == 0:
                continue
            
            # Particles clustered around node
            spread = R * 0.15 if i in [0, num_nodes-1] else R * 0.08
            positions[mask, 0] = node_positions[i] + np.random.normal(0, spread, node_n)
            positions[mask, 1] = np.random.normal(0, spread * 0.5, node_n)
            positions[mask, 2] = np.random.normal(0, spread * 0.5, node_n)
        
        # Low random velocities
        velocities = np.random.normal(0, 0.5, (n, 3))
        
    elif distribution == "explosion":
        # Expanding supernova / explosion
        phi = np.random.uniform(0, 2 * np.pi, n)
        cos_theta = np.random.uniform(-1, 1, n)
        sin_theta = np.sqrt(1 - cos_theta**2)
        
        r = np.random.exponential(R * 0.1, n)
        
        positions[:, 0] = r * sin_theta * np.cos(phi)
        positions[:, 1] = r * cos_theta
        positions[:, 2] = r * sin_theta * np.sin(phi)
        
        # Radial outward velocity
        speed = 10.0 + np.random.exponential(5.0, n)
        norm = np.sqrt(np.sum(positions**2, axis=1, keepdims=True)) + 0.01
        velocities = positions / norm * speed[:, np.newaxis]
        
    elif distribution == "vortex":
        # Swirling vortex
        r = np.random.exponential(R * 0.3, n)
        theta = np.random.uniform(0, 2 * np.pi, n)
        z = np.random.normal(0, R * 0.1, n)
        
        positions[:, 0] = r * np.cos(theta)
        positions[:, 1] = z
        positions[:, 2] = r * np.sin(theta)
        
        # Strong tangential + upward velocity
        tangent_speed = 8.0 / (r / R + 0.2)
        velocities[:, 0] = -tangent_speed * np.sin(theta)
        velocities[:, 2] = tangent_speed * np.cos(theta)
        velocities[:, 1] = 2.0 * np.sign(z)  # Upward/downward spiral
        
    elif distribution == "cube":
        # Cubic lattice (for testing)
        side = int(np.ceil(n ** (1/3)))
        grid = np.mgrid[0:side, 0:side, 0:side].reshape(3, -1).T
        grid = grid[:n]
        
        spacing = R * 2 / side
        positions[:, :] = (grid - side / 2) * spacing
        velocities = np.random.normal(0, 0.1, (n, 3))
        
    elif distribution == "pleiades":
        # Young star cluster with nebulosity
        # Core cluster
        core_n = n // 5
        nebula_n = n - core_n
        
        # Dense core
        r_core = np.random.exponential(R * 0.1, core_n)
        phi_core = np.random.uniform(0, 2 * np.pi, core_n)
        cos_theta_core = np.random.uniform(-1, 1, core_n)
        sin_theta_core = np.sqrt(1 - cos_theta_core**2)
        
        positions[:core_n, 0] = r_core * sin_theta_core * np.cos(phi_core)
        positions[:core_n, 1] = r_core * cos_theta_core
        positions[:core_n, 2] = r_core * sin_theta_core * np.sin(phi_core)
        masses[:core_n] = 5.0  # Brighter stars
        
        # Surrounding nebula
        r_neb = np.random.exponential(R * 0.5, nebula_n) + R * 0.1
        phi_neb = np.random.uniform(0, 2 * np.pi, nebula_n)
        cos_theta_neb = np.random.uniform(-1, 1, nebula_n)
        sin_theta_neb = np.sqrt(1 - cos_theta_neb**2)
        
        positions[core_n:, 0] = r_neb * sin_theta_neb * np.cos(phi_neb)
        positions[core_n:, 1] = r_neb * cos_theta_neb * 0.5  # Flattened
        positions[core_n:, 2] = r_neb * sin_theta_neb * np.sin(phi_neb)
        
        # Small velocities
        sigma = np.sqrt(G * core_n * 5 * 0.001 / (R * 0.2))
        velocities = np.random.normal(0, sigma * 0.5, (n, 3))
        
    else:  # sphere / default
        phi = np.random.uniform(0, 2 * np.pi, n)
        cos_theta = np.random.uniform(-1, 1, n)
        sin_theta = np.sqrt(1 - cos_theta**2)
        r = np.random.uniform(0, R, n) ** (1/3) * R  # Uniform in volume
        
        positions[:, 0] = r * sin_theta * np.cos(phi)
        positions[:, 1] = r * cos_theta
        positions[:, 2] = r * sin_theta * np.sin(phi)
        
        velocities = np.random.normal(0, 0.5, (n, 3))
    
    return positions, velocities, masses


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

PRESETS: Dict[str, dict] = {}

# -----------------------------------------------------------------------------
# CINEMATIC PRESETS (Beautiful, production-quality)
# -----------------------------------------------------------------------------

PRESETS["galaxy_epic"] = {
    "name": "Epic Galaxy",
    "description": "Massive spiral galaxy, cinematic quality",
    "category": "CINEMATIC",
    "num_bodies": 500_000,
    "theta": 0.7,
    "G": 0.1,
    "softening": 2.5,
    "damping": 1.0,
    "spawn_radius": 600.0,
    "distribution": "galaxy",
    "total_frames": 3000,
    "dt_per_frame": 0.12,
    "substeps": 3,
    "target_fps": 24,
    "estimated_time": "~2 hours",
}

PRESETS["collision_majesty"] = {
    "name": "Galactic Collision",
    "description": "Two massive galaxies colliding, Andromeda-style",
    "category": "CINEMATIC",
    "num_bodies": 400_000,
    "theta": 0.75,
    "G": 0.12,
    "softening": 2.0,
    "damping": 1.0,
    "spawn_radius": 700.0,
    "distribution": "collision",
    "total_frames": 4000,
    "dt_per_frame": 0.15,
    "substeps": 3,
    "target_fps": 24,
    "estimated_time": "~3 hours",
}

PRESETS["spiral_milkyway"] = {
    "name": "Milky Way Spiral",
    "description": "Four-arm spiral galaxy like our Milky Way",
    "category": "CINEMATIC",
    "num_bodies": 300_000,
    "theta": 0.8,
    "G": 0.08,
    "softening": 2.0,
    "damping": 1.0,
    "spawn_radius": 600.0,
    "distribution": "spiral",
    "total_frames": 2500,
    "dt_per_frame": 0.1,
    "substeps": 3,
    "target_fps": 24,
    "estimated_time": "~1.5 hours",
}

PRESETS["bar_galaxy"] = {
    "name": "Barred Spiral Galaxy",
    "description": "Galaxy with central bar structure, like SBb type",
    "category": "CINEMATIC",
    "num_bodies": 350_000,
    "theta": 0.8,
    "G": 0.09,
    "softening": 2.0,
    "damping": 1.0,
    "spawn_radius": 550.0,
    "distribution": "bar",
    "total_frames": 2000,
    "dt_per_frame": 0.12,
    "substeps": 3,
    "target_fps": 24,
    "estimated_time": "~1.5 hours",
}

# -----------------------------------------------------------------------------
# FAST PRESETS (Quick renders, good for testing)
# -----------------------------------------------------------------------------

PRESETS["quick_galaxy"] = {
    "name": "Quick Galaxy",
    "description": "Fast galaxy simulation for testing",
    "category": "FAST",
    "num_bodies": 100_000,
    "theta": 0.95,
    "G": 0.15,
    "softening": 3.0,
    "damping": 1.0,
    "spawn_radius": 500.0,
    "distribution": "galaxy",
    "total_frames": 500,
    "dt_per_frame": 0.2,
    "substeps": 1,
    "target_fps": 30,
    "estimated_time": "~3 minutes",
}

PRESETS["quick_collision"] = {
    "name": "Quick Collision",
    "description": "Fast collision simulation",
    "category": "FAST",
    "num_bodies": 80_000,
    "theta": 0.95,
    "G": 0.2,
    "softening": 3.5,
    "damping": 1.0,
    "spawn_radius": 400.0,
    "distribution": "collision",
    "total_frames": 600,
    "dt_per_frame": 0.25,
    "substeps": 1,
    "target_fps": 30,
    "estimated_time": "~4 minutes",
}

PRESETS["mini_cluster"] = {
    "name": "Mini Cluster",
    "description": "Small dense star cluster",
    "category": "FAST",
    "num_bodies": 50_000,
    "theta": 0.95,
    "G": 0.2,
    "softening": 2.0,
    "damping": 1.0,
    "spawn_radius": 200.0,
    "distribution": "cluster",
    "total_frames": 400,
    "dt_per_frame": 0.15,
    "substeps": 1,
    "target_fps": 30,
    "estimated_time": "~2 minutes",
}

PRESETS["instant_ring"] = {
    "name": "Instant Ring",
    "description": "Saturn-like ring, very fast",
    "category": "FAST",
    "num_bodies": 60_000,
    "theta": 0.95,
    "G": 0.1,
    "softening": 2.0,
    "damping": 1.0,
    "spawn_radius": 300.0,
    "distribution": "ring",
    "total_frames": 300,
    "dt_per_frame": 0.2,
    "substeps": 1,
    "target_fps": 30,
    "estimated_time": "~1 minute",
}

# -----------------------------------------------------------------------------
# SCIENTIFIC PRESETS (Physically accurate)
# -----------------------------------------------------------------------------

PRESETS["accurate_cluster"] = {
    "name": "Globular Cluster",
    "description": "Physically accurate globular cluster (Plummer model)",
    "category": "SCIENTIFIC",
    "num_bodies": 200_000,
    "theta": 0.5,  # Very accurate
    "G": 0.05,
    "softening": 1.0,
    "damping": 1.0,
    "spawn_radius": 300.0,
    "distribution": "cluster",
    "total_frames": 2000,
    "dt_per_frame": 0.08,
    "substeps": 4,
    "target_fps": 24,
    "estimated_time": "~4 hours",
}

PRESETS["elliptical_galaxy"] = {
    "name": "Elliptical Galaxy",
    "description": "Giant elliptical galaxy (E3 type)",
    "category": "SCIENTIFIC",
    "num_bodies": 250_000,
    "theta": 0.6,
    "G": 0.06,
    "softening": 2.0,
    "damping": 1.0,
    "spawn_radius": 500.0,
    "distribution": "elliptical",
    "total_frames": 2000,
    "dt_per_frame": 0.1,
    "substeps": 3,
    "target_fps": 24,
    "estimated_time": "~3 hours",
}

PRESETS["binary_stars"] = {
    "name": "Binary Star System",
    "description": "Two stars with protoplanetary disks",
    "category": "SCIENTIFIC",
    "num_bodies": 150_000,
    "theta": 0.7,
    "G": 0.15,
    "softening": 1.5,
    "damping": 1.0,
    "spawn_radius": 400.0,
    "distribution": "binary",
    "total_frames": 1500,
    "dt_per_frame": 0.1,
    "substeps": 3,
    "target_fps": 24,
    "estimated_time": "~2 hours",
}

PRESETS["tidal_stream"] = {
    "name": "Tidal Stream",
    "description": "Stellar stream from disrupted dwarf galaxy",
    "category": "SCIENTIFIC",
    "num_bodies": 100_000,
    "theta": 0.8,
    "G": 0.05,
    "softening": 2.0,
    "damping": 1.0,
    "spawn_radius": 800.0,
    "distribution": "stream",
    "total_frames": 1200,
    "dt_per_frame": 0.15,
    "substeps": 2,
    "target_fps": 24,
    "estimated_time": "~1 hour",
}

# -----------------------------------------------------------------------------
# CHAOS PRESETS (Wild, unpredictable)
# -----------------------------------------------------------------------------

PRESETS["supernova"] = {
    "name": "Supernova Explosion",
    "description": "Violent expanding shell from stellar explosion",
    "category": "CHAOS",
    "num_bodies": 150_000,
    "theta": 0.9,
    "G": 0.05,
    "softening": 1.0,
    "damping": 0.99,
    "spawn_radius": 100.0,
    "distribution": "explosion",
    "total_frames": 1000,
    "dt_per_frame": 0.1,
    "substeps": 2,
    "target_fps": 30,
    "estimated_time": "~30 minutes",
}

PRESETS["cosmic_vortex"] = {
    "name": "Cosmic Vortex",
    "description": "Swirling maelstrom of stars",
    "category": "CHAOS",
    "num_bodies": 200_000,
    "theta": 0.9,
    "G": 0.08,
    "softening": 2.0,
    "damping": 0.995,
    "spawn_radius": 400.0,
    "distribution": "vortex",
    "total_frames": 1500,
    "dt_per_frame": 0.12,
    "substeps": 2,
    "target_fps": 30,
    "estimated_time": "~1 hour",
}

PRESETS["triple_collision"] = {
    "name": "Triple Collision",
    "description": "Three galaxies colliding chaotically",
    "category": "CHAOS",
    "num_bodies": 300_000,
    "theta": 0.85,
    "G": 0.15,
    "softening": 2.5,
    "damping": 1.0,
    "spawn_radius": 600.0,
    "distribution": "filament",  # Creates multi-node structure
    "total_frames": 2000,
    "dt_per_frame": 0.15,
    "substeps": 2,
    "target_fps": 24,
    "estimated_time": "~1.5 hours",
}

PRESETS["gravity_bomb"] = {
    "name": "Gravity Bomb",
    "description": "Uniform sphere collapsing violently",
    "category": "CHAOS",
    "num_bodies": 200_000,
    "theta": 0.9,
    "G": 0.3,  # Very strong gravity
    "softening": 1.0,
    "damping": 1.0,
    "spawn_radius": 500.0,
    "distribution": "sphere",
    "total_frames": 800,
    "dt_per_frame": 0.1,
    "substeps": 2,
    "target_fps": 30,
    "estimated_time": "~30 minutes",
}

# -----------------------------------------------------------------------------
# ARTISTIC PRESETS (Visually striking)
# -----------------------------------------------------------------------------

PRESETS["nebula_birth"] = {
    "name": "Star Cluster Birth",
    "description": "Young star cluster emerging from nebula",
    "category": "ARTISTIC",
    "num_bodies": 250_000,
    "theta": 0.85,
    "G": 0.08,
    "softening": 2.0,
    "damping": 1.0,
    "spawn_radius": 500.0,
    "distribution": "pleiades",
    "total_frames": 1500,
    "dt_per_frame": 0.12,
    "substeps": 2,
    "target_fps": 24,
    "estimated_time": "~1 hour",
}

PRESETS["saturn_rings"] = {
    "name": "Saturn's Rings",
    "description": "Beautiful ring system with dense core",
    "category": "ARTISTIC",
    "num_bodies": 300_000,
    "theta": 0.85,
    "G": 0.08,
    "softening": 1.5,
    "damping": 1.0,
    "spawn_radius": 400.0,
    "distribution": "ring",
    "total_frames": 1500,
    "dt_per_frame": 0.1,
    "substeps": 2,
    "target_fps": 24,
    "estimated_time": "~1 hour",
}

PRESETS["shell_collapse"] = {
    "name": "Shell Collapse",
    "description": "Hollow shell collapsing inward",
    "category": "ARTISTIC",
    "num_bodies": 200_000,
    "theta": 0.85,
    "G": 0.15,
    "softening": 2.0,
    "damping": 1.0,
    "spawn_radius": 400.0,
    "distribution": "shell",
    "total_frames": 1200,
    "dt_per_frame": 0.12,
    "substeps": 2,
    "target_fps": 24,
    "estimated_time": "~45 minutes",
}

PRESETS["cosmic_web"] = {
    "name": "Cosmic Web",
    "description": "Large-scale structure of the universe",
    "category": "ARTISTIC",
    "num_bodies": 200_000,
    "theta": 0.9,
    "G": 0.06,
    "softening": 3.0,
    "damping": 1.0,
    "spawn_radius": 800.0,
    "distribution": "filament",
    "total_frames": 1500,
    "dt_per_frame": 0.15,
    "substeps": 2,
    "target_fps": 24,
    "estimated_time": "~1 hour",
}

# -----------------------------------------------------------------------------
# MEGA PRESETS (Very large, long renders)
# -----------------------------------------------------------------------------

PRESETS["million_stars"] = {
    "name": "Million Star Galaxy",
    "description": "Massive 1M body galaxy (very long render)",
    "category": "MEGA",
    "num_bodies": 1_000_000,
    "theta": 0.95,
    "G": 0.1,
    "softening": 3.0,
    "damping": 1.0,
    "spawn_radius": 800.0,
    "distribution": "galaxy",
    "total_frames": 2000,
    "dt_per_frame": 0.15,
    "substeps": 2,
    "target_fps": 24,
    "estimated_time": "~6 hours",
}

PRESETS["mega_collision"] = {
    "name": "Mega Collision",
    "description": "Two 500K body galaxies colliding",
    "category": "MEGA",
    "num_bodies": 1_000_000,
    "theta": 0.95,
    "G": 0.12,
    "softening": 3.5,
    "damping": 1.0,
    "spawn_radius": 1000.0,
    "distribution": "collision",
    "total_frames": 3000,
    "dt_per_frame": 0.15,
    "substeps": 2,
    "target_fps": 24,
    "estimated_time": "~10 hours",
}

# -----------------------------------------------------------------------------
# TINY PRESETS (For testing on slow machines)
# -----------------------------------------------------------------------------

PRESETS["tiny_galaxy"] = {
    "name": "Tiny Galaxy",
    "description": "Very small galaxy for testing",
    "category": "TINY",
    "num_bodies": 10_000,
    "theta": 0.95,
    "G": 0.2,
    "softening": 5.0,
    "damping": 1.0,
    "spawn_radius": 200.0,
    "distribution": "galaxy",
    "total_frames": 200,
    "dt_per_frame": 0.3,
    "substeps": 1,
    "target_fps": 30,
    "estimated_time": "~30 seconds",
}

PRESETS["tiny_collision"] = {
    "name": "Tiny Collision",
    "description": "Very small collision for testing",
    "category": "TINY",
    "num_bodies": 15_000,
    "theta": 0.95,
    "G": 0.25,
    "softening": 5.0,
    "damping": 1.0,
    "spawn_radius": 250.0,
    "distribution": "collision",
    "total_frames": 250,
    "dt_per_frame": 0.3,
    "substeps": 1,
    "target_fps": 30,
    "estimated_time": "~45 seconds",
}

PRESETS["demo_cluster"] = {
    "name": "Demo Cluster",
    "description": "Quick demo of cluster dynamics",
    "category": "TINY",
    "num_bodies": 20_000,
    "theta": 0.95,
    "G": 0.15,
    "softening": 3.0,
    "damping": 1.0,
    "spawn_radius": 150.0,
    "distribution": "cluster",
    "total_frames": 300,
    "dt_per_frame": 0.2,
    "substeps": 1,
    "target_fps": 30,
    "estimated_time": "~1 minute",
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_preset_list() -> List[Tuple[str, dict]]:
    """Get list of all presets sorted by category."""
    category_order = ["TINY", "FAST", "CINEMATIC", "ARTISTIC", "SCIENTIFIC", "CHAOS", "MEGA"]
    
    sorted_presets = sorted(
        PRESETS.items(),
        key=lambda x: (category_order.index(x[1]["category"]) if x[1]["category"] in category_order else 99, x[0])
    )
    return sorted_presets


def print_preset_menu():
    """Print formatted preset selection menu."""
    presets = get_preset_list()
    current_category = None
    
    print("\n" + "=" * 70)
    print("  N-BODY SIMULATION RECORDING PRESETS")
    print("=" * 70)
    
    for idx, (key, preset) in enumerate(presets):
        if preset["category"] != current_category:
            current_category = preset["category"]
            print(f"\n{'─' * 70}")
            print(f"  {current_category}")
            print(f"{'─' * 70}")
        
        bodies = preset["num_bodies"]
        if bodies >= 1_000_000:
            bodies_str = f"{bodies / 1_000_000:.1f}M"
        else:
            bodies_str = f"{bodies // 1000}K"
        
        frames = preset["total_frames"]
        time_est = preset.get("estimated_time", "?")
        
        print(f"  [{idx:2d}] {preset['name']:<25} {bodies_str:>6} bodies | {frames:>4} frames | {time_est}")
        print(f"       {preset['description']}")
    
    print(f"\n{'=' * 70}")
    print("  Enter number [0-{}] to select, or 'q' to quit".format(len(presets) - 1))
    print("=" * 70)


def get_preset_by_index(index: int) -> Tuple[str, dict]:
    """Get preset by menu index."""
    presets = get_preset_list()
    if 0 <= index < len(presets):
        return presets[index]
    return None, None


def get_preset_config(key: str) -> dict:
    """Get a preset configuration by key, adding session_name."""
    if key not in PRESETS:
        return None
    
    preset = PRESETS[key].copy()
    # Generate session name from key
    preset["session_name"] = key
    return preset


def list_distributions():
    """Print all available distributions."""
    print("\nAvailable spawn distributions:")
    print("-" * 40)
    for name, desc in DISTRIBUTIONS.items():
        print(f"  {name:<15} - {desc}")

