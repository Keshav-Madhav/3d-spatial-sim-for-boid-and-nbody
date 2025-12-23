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
    "double_helix": "DNA-like double helix structure",
    "accretion_disk": "Black hole accretion disk with jets",
    "torus": "Donut-shaped torus",
    "hourglass": "Binary star hourglass nebula",
    "fibonacci": "Fibonacci spiral pattern",
    "triple": "Three galaxies in triangle formation",
    "rosette": "Flower-like orbital rosette",
    "dyson": "Dyson sphere megastructure",
}


def compute_rotation_curve(r: np.ndarray, masses: np.ndarray, G: float, softening: float) -> np.ndarray:
    """
    Compute proper circular velocity for a softened self-gravitating disk.
    
    Uses sorted enclosed mass with softening that matches the N-body softening.
    The rotation curve smoothly goes to zero at r=0 (no singularity).
    """
    n = len(r)
    
    # Sort by radius
    sort_idx = np.argsort(r)
    sorted_r = r[sort_idx]
    sorted_masses = masses[sort_idx]
    
    # Cumulative mass (enclosed mass)
    cumulative_mass = np.cumsum(sorted_masses)
    
    # For stability, use a Plummer-like rotation curve:
    # v_c = sqrt(G * M_enc * r^2 / (r^2 + eps^2)^(3/2))
    # This smoothly goes to 0 at r=0 and approaches Keplerian at large r
    eps = softening * 2  # Use larger effective softening for stability
    eps_sq = eps ** 2
    r_sq = sorted_r ** 2
    
    # Rotation velocity with proper softening
    v_circular = np.sqrt(G * cumulative_mass * r_sq / (r_sq + eps_sq) ** 1.5)
    
    # Aggressive inner damping - use sigmoid-like curve
    # This ensures center particles have near-zero velocity
    inner_scale = softening * 3  # Damping scale
    inner_damping = (sorted_r ** 2) / (sorted_r ** 2 + inner_scale ** 2)
    v_circular *= inner_damping
    
    # Map back to original order
    inverse_idx = np.argsort(sort_idx)
    return v_circular[inverse_idx]


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
        # Stable exponential disk galaxy with proper rotation curve
        scale_length = R * 0.3  # Exponential scale length
        softening = R * 0.03  # Match typical N-body softening
        
        # Generate radii from exponential distribution (no hard clip!)
        r = np.random.exponential(scale_length, n)
        
        # Soft truncation: smoothly reduce density at large radii
        # instead of hard clipping which creates edge artifacts
        max_r = R * 1.2
        r = r * (1 - np.exp(-max_r / (r + 0.01)))  # Soft cap
        r = np.maximum(r, R * 0.001)  # Just prevent exactly zero
        
        theta = np.random.uniform(0, 2 * np.pi, n)
        
        # Thin disk with slight flare at edges
        disk_height = R * 0.012 * (1 + (r / R) ** 0.5 * 0.3)
        z = np.random.normal(0, 1, n) * disk_height
        
        positions[:, 0] = r * np.cos(theta)
        positions[:, 1] = z
        positions[:, 2] = r * np.sin(theta)
        
        # Compute proper rotation curve with softening
        orbital_speed = compute_rotation_curve(r, masses, G, softening)
        
        # Tangential velocity (counter-clockwise in XZ plane)
        velocities[:, 0] = -orbital_speed * np.sin(theta)
        velocities[:, 2] = orbital_speed * np.cos(theta)
        
        # Velocity dispersion (Toomre stability)
        # Scale dispersion with radius - very small at center, larger at edge
        radial_factor = r / (r + softening * 2)  # 0 at center, ~1 at large r
        sigma = orbital_speed * 0.10 * radial_factor + np.sqrt(G * n * 0.00005)
        velocities[:, 0] += np.random.normal(0, sigma, n)
        velocities[:, 2] += np.random.normal(0, sigma, n)
        velocities[:, 1] = np.random.normal(0, sigma * 0.25, n)
        
    elif distribution == "collision":
        # Two stable galaxies on collision course
        half = n // 2
        n2 = n - half
        scale_length = R * 0.25
        softening = R * 0.025
        
        # ===== Galaxy 1 (centered at -R*0.7) =====
        r1 = np.random.exponential(scale_length, half)
        # Soft truncation (no hard clip!)
        max_r1 = R * 0.6
        r1 = r1 * (1 - np.exp(-max_r1 / (r1 + 0.01)))
        r1 = np.maximum(r1, R * 0.001)
        
        theta1 = np.random.uniform(0, 2 * np.pi, half)
        
        # Positions (galaxy 1 at left)
        galaxy1_x = -R * 0.7
        positions[:half, 0] = r1 * np.cos(theta1) + galaxy1_x
        disk_height1 = R * 0.01 * (1 + (r1 / R) ** 0.5 * 0.3)
        positions[:half, 1] = np.random.normal(0, 1, half) * disk_height1
        positions[:half, 2] = r1 * np.sin(theta1)
        
        # Proper rotation curve for galaxy 1
        masses1 = masses[:half]
        orbital_speed1 = compute_rotation_curve(r1, masses1, G, softening)
        
        # Tangential velocity (counter-clockwise)
        velocities[:half, 0] = -orbital_speed1 * np.sin(theta1)
        velocities[:half, 2] = orbital_speed1 * np.cos(theta1)
        
        # Velocity dispersion - scales with radius
        radial_factor1 = r1 / (r1 + softening * 2)
        sigma1 = orbital_speed1 * 0.10 * radial_factor1 + np.sqrt(G * half * 0.00005)
        velocities[:half, 0] += np.random.normal(0, sigma1, half)
        velocities[:half, 2] += np.random.normal(0, sigma1, half)
        velocities[:half, 1] = np.random.normal(0, sigma1 * 0.25, half)
        
        # Bulk motion toward galaxy 2
        collision_speed = 1.2
        velocities[:half, 0] += collision_speed
        
        # ===== Galaxy 2 (centered at +R*0.7, offset in Y) =====
        r2 = np.random.exponential(scale_length, n2)
        # Soft truncation
        max_r2 = R * 0.6
        r2 = r2 * (1 - np.exp(-max_r2 / (r2 + 0.01)))
        r2 = np.maximum(r2, R * 0.001)
        
        theta2 = np.random.uniform(0, 2 * np.pi, n2)
        
        # Positions (galaxy 2 at right, offset in Y for off-center collision)
        galaxy2_x = R * 0.7
        galaxy2_y = R * 0.12
        positions[half:, 0] = r2 * np.cos(theta2) + galaxy2_x
        disk_height2 = R * 0.01 * (1 + (r2 / R) ** 0.5 * 0.3)
        positions[half:, 1] = np.random.normal(0, 1, n2) * disk_height2 + galaxy2_y
        positions[half:, 2] = r2 * np.sin(theta2)
        
        # Proper rotation curve for galaxy 2
        masses2 = masses[half:]
        orbital_speed2 = compute_rotation_curve(r2, masses2, G, softening)
        
        # Tangential velocity (clockwise - opposite spin for interesting dynamics)
        velocities[half:, 0] = orbital_speed2 * np.sin(theta2)
        velocities[half:, 2] = -orbital_speed2 * np.cos(theta2)
        
        # Velocity dispersion - scales with radius
        radial_factor2 = r2 / (r2 + softening * 2)
        sigma2 = orbital_speed2 * 0.10 * radial_factor2 + np.sqrt(G * n2 * 0.00005)
        velocities[half:, 0] += np.random.normal(0, sigma2, n2)
        velocities[half:, 2] += np.random.normal(0, sigma2, n2)
        velocities[half:, 1] = np.random.normal(0, sigma2 * 0.25, n2)
        
        # Bulk motion toward galaxy 1
        velocities[half:, 0] -= collision_speed
        
    elif distribution == "spiral":
        # Stable multi-arm spiral galaxy with proper rotation
        scale_length = R * 0.3
        softening = R * 0.03
        
        # Exponential radial distribution (no hard clip!)
        disk_r = np.random.exponential(scale_length, n)
        
        # Soft truncation
        max_r = R * 1.2
        disk_r = disk_r * (1 - np.exp(-max_r / (disk_r + 0.01)))
        disk_r = np.maximum(disk_r, R * 0.001)
        
        # Logarithmic spiral pattern
        spiral_tightness = 0.35
        num_arms = 4
        
        base_theta = np.log(disk_r / (R * 0.02) + 1) / spiral_tightness
        arm_assignment = np.random.randint(0, num_arms, n)
        arm_offset = arm_assignment * (2 * np.pi / num_arms)
        
        # Scatter increases with radius (arms are tighter near center)
        arm_scatter = 0.12 + 0.15 * (disk_r / R) ** 0.5
        theta = base_theta + arm_offset + np.random.normal(0, arm_scatter, n)
        
        # Positions
        positions[:, 0] = disk_r * np.cos(theta)
        positions[:, 2] = disk_r * np.sin(theta)
        
        # Disk height
        disk_height = R * 0.012 * (1 + (disk_r / R) ** 0.5 * 0.3)
        positions[:, 1] = np.random.normal(0, 1, n) * disk_height
        
        # Proper rotation curve
        orbital_speed = compute_rotation_curve(disk_r, masses, G, softening)
        
        # Tangential velocity (perpendicular to radius vector)
        # Use actual position angle, not spiral angle
        pos_theta = np.arctan2(positions[:, 2], positions[:, 0])
        velocities[:, 0] = -orbital_speed * np.sin(pos_theta)
        velocities[:, 2] = orbital_speed * np.cos(pos_theta)
        
        # Velocity dispersion - scales with radius (small at center)
        radial_factor = disk_r / (disk_r + softening * 2)
        sigma = orbital_speed * 0.10 * radial_factor + np.sqrt(G * n * 0.00005)
        velocities[:, 0] += np.random.normal(0, sigma, n)
        velocities[:, 2] += np.random.normal(0, sigma, n)
        velocities[:, 1] = np.random.normal(0, sigma * 0.25, n)
        
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
        # Cosmic Web - Large scale structure like CMBR
        # Best with 50-100M bodies for proper large-scale structure
        
        # Create a 3D grid of potential cluster centers
        grid_size = 8  # 8x8x8 grid = 512 potential nodes
        node_spacing = R * 2.5 / grid_size  # Spread over large volume
        
        # Generate grid positions
        grid_coords = np.linspace(-R * 1.25, R * 1.25, grid_size)
        node_centers = []
        for ix in range(grid_size):
            for iy in range(grid_size):
                for iz in range(grid_size):
                    node_centers.append([
                        grid_coords[ix],
                        grid_coords[iy],
                        grid_coords[iz]
                    ])
        node_centers = np.array(node_centers)
        
        # Randomly select which nodes will be "active" (have matter)
        # Only ~30-40% of nodes are active (creates voids)
        num_nodes = len(node_centers)
        active_prob = 0.35
        active_nodes = np.random.random(num_nodes) < active_prob
        active_centers = node_centers[active_nodes]
        
        # Assign each particle to a random active node
        # With power-law distribution (some nodes are denser)
        num_active = len(active_centers)
        node_weights = np.random.power(2.0, num_active)  # Power law
        node_weights /= node_weights.sum()
        
        particle_nodes = np.random.choice(
            num_active, 
            size=n, 
            p=node_weights
        )
        
        # Place particles around their assigned nodes
        for node_idx in range(num_active):
            mask = particle_nodes == node_idx
            node_n = np.sum(mask)
            if node_n == 0:
                continue
            
            center = active_centers[node_idx]
            
            # Filamentary structure: elongated along random axis
            # Each node extends along 1-3 random directions to neighbors
            elongation_axis = np.random.randn(3)
            elongation_axis /= (np.linalg.norm(elongation_axis) + 1e-10)
            
            # Particles form elongated structure
            # High spread along elongation, low perpendicular
            parallel_offset = np.random.normal(0, node_spacing * 0.8, node_n)
            
            # Perpendicular offset is much smaller (filament thickness)
            perp_vector1 = np.random.randn(3)
            perp_vector1 -= perp_vector1.dot(elongation_axis) * elongation_axis
            perp_vector1 /= (np.linalg.norm(perp_vector1) + 1e-10)
            
            perp_vector2 = np.cross(elongation_axis, perp_vector1)
            perp_vector2 /= (np.linalg.norm(perp_vector2) + 1e-10)
            
            perp_offset1 = np.random.normal(0, node_spacing * 0.12, node_n)
            perp_offset2 = np.random.normal(0, node_spacing * 0.12, node_n)
            
            # Combine offsets
            for i, pidx in enumerate(np.where(mask)[0]):
                positions[pidx] = (
                    center + 
                    parallel_offset[i] * elongation_axis +
                    perp_offset1[i] * perp_vector1 +
                    perp_offset2[i] * perp_vector2
                )
        
        # Add very small "Hubble flow" - expansion
        distances = np.linalg.norm(positions, axis=1, keepdims=True)
        hubble_constant = 0.05
        velocities = positions * hubble_constant + np.random.normal(0, 0.3, (n, 3))
        
        # Very low mass for individual particles (many particles per structure)
        masses[:] = 0.1
        
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
        
    elif distribution == "double_helix":
        # DNA-like double helix structure
        t = np.linspace(0, 6 * np.pi, n)  # More turns for more dramatic helix
        radius = R * 0.25  # Slightly tighter radius
        pitch = R * 2.0  # Much taller! (was 0.5)
        
        # Split into two helices
        half = n // 2
        
        # Helix 1
        positions[:half, 0] = radius * np.cos(t[:half])
        positions[:half, 1] = (t[:half] / (6 * np.pi)) * pitch - pitch/2
        positions[:half, 2] = radius * np.sin(t[:half])
        
        # Helix 2 (offset by 180 degrees)
        positions[half:, 0] = radius * np.cos(t[half:] + np.pi)
        positions[half:, 1] = (t[half:] / (6 * np.pi)) * pitch - pitch/2
        positions[half:, 2] = radius * np.sin(t[half:] + np.pi)
        
        # Add minimal noise to maintain structure
        positions += np.random.normal(0, R * 0.01, (n, 3))
        
        # Much slower rotation to maintain shape
        omega = 0.08  # Reduced from 0.5
        # Rotate around the helical axis (Y-axis)
        velocities[:half, 0] = -omega * positions[:half, 2]
        velocities[:half, 2] = omega * positions[:half, 0]
        velocities[half:, 0] = -omega * positions[half:, 2]
        velocities[half:, 2] = omega * positions[half:, 0]
        
        # Add slight vertical motion along the helix
        velocities[:, 1] = np.random.normal(0, omega * 0.3, n)
        
    elif distribution == "accretion_disk":
        # Black hole accretion disk with polar jets
        disk_n = int(n * 0.85)
        jet_n = n - disk_n
        
        # Central supermassive black hole (invisible, just mass)
        central_mass = 1000.0
        
        # Accretion disk
        r_disk = np.random.exponential(R * 0.2, disk_n)
        r_disk = np.clip(r_disk, R * 0.05, R * 0.8)
        theta_disk = np.random.uniform(0, 2 * np.pi, disk_n)
        
        # Very thin disk
        z_disk = np.random.normal(0, R * 0.01, disk_n)
        
        positions[:disk_n, 0] = r_disk * np.cos(theta_disk)
        positions[:disk_n, 1] = z_disk
        positions[:disk_n, 2] = r_disk * np.sin(theta_disk)
        
        # Keplerian rotation around central mass
        v_kep = np.sqrt(G * central_mass / (r_disk + R * 0.05))
        velocities[:disk_n, 0] = -v_kep * np.sin(theta_disk)
        velocities[:disk_n, 2] = v_kep * np.cos(theta_disk)
        
        # Jets (bipolar outflow)
        if jet_n > 0:
            jet_half = jet_n // 2
            jet_lower_n = jet_n - jet_half
            
            # Upper jet
            z_jet_up = np.random.uniform(R * 0.2, R * 1.2, jet_half)
            r_jet_up = np.random.exponential(R * 0.05, jet_half)
            theta_jet_up = np.random.uniform(0, 2 * np.pi, jet_half)
            
            positions[disk_n:disk_n+jet_half, 0] = r_jet_up * np.cos(theta_jet_up)
            positions[disk_n:disk_n+jet_half, 1] = z_jet_up
            positions[disk_n:disk_n+jet_half, 2] = r_jet_up * np.sin(theta_jet_up)
            velocities[disk_n:disk_n+jet_half, 1] = 3.0  # Outflow velocity
            
            # Lower jet (generate new arrays for correct size)
            z_jet_down = np.random.uniform(R * 0.2, R * 1.2, jet_lower_n)
            r_jet_down = np.random.exponential(R * 0.05, jet_lower_n)
            theta_jet_down = np.random.uniform(0, 2 * np.pi, jet_lower_n)
            
            positions[disk_n+jet_half:, 0] = r_jet_down * np.cos(theta_jet_down)
            positions[disk_n+jet_half:, 1] = -z_jet_down
            positions[disk_n+jet_half:, 2] = r_jet_down * np.sin(theta_jet_down)
            velocities[disk_n+jet_half:, 1] = -3.0
        
        masses[:disk_n] = 0.5
        masses[disk_n:] = 0.1  # Jets are less massive
        
    elif distribution == "torus":
        # Donut shape - particles orbit around a ring
        major_radius = R * 0.6  # Distance to center of tube
        minor_radius = R * 0.25  # Tube radius
        
        # Parametric torus
        u = np.random.uniform(0, 2 * np.pi, n)  # Around tube
        v = np.random.uniform(0, 2 * np.pi, n)  # Around major circle
        
        # Add some noise for thickness
        r_noise = np.random.normal(1.0, 0.1, n)
        
        positions[:, 0] = (major_radius + minor_radius * np.cos(u) * r_noise) * np.cos(v)
        positions[:, 1] = minor_radius * np.sin(u) * r_noise
        positions[:, 2] = (major_radius + minor_radius * np.cos(u) * r_noise) * np.sin(v)
        
        # Orbital velocity around major axis
        omega = np.sqrt(G * n * 0.001 / major_radius)
        velocities[:, 0] = -omega * positions[:, 2]
        velocities[:, 2] = omega * positions[:, 0]
        
        # Small random motion
        velocities += np.random.normal(0, omega * 0.1, (n, 3))
        
    elif distribution == "hourglass":
        # Binary star hourglass nebula (two cones meeting at tips)
        half = n // 2
        
        # Upper cone
        z_up = np.random.uniform(0, R, half)
        r_up = z_up * 0.5 * (1 + np.random.normal(0, 0.1, half))
        theta_up = np.random.uniform(0, 2 * np.pi, half)
        
        positions[:half, 0] = r_up * np.cos(theta_up)
        positions[:half, 1] = z_up
        positions[:half, 2] = r_up * np.sin(theta_up)
        velocities[:half, 1] = 1.5  # Expanding upward
        
        # Lower cone
        z_down = np.random.uniform(-R, 0, n - half)
        r_down = -z_down * 0.5 * (1 + np.random.normal(0, 0.1, n - half))
        theta_down = np.random.uniform(0, 2 * np.pi, n - half)
        
        positions[half:, 0] = r_down * np.cos(theta_down)
        positions[half:, 1] = z_down
        positions[half:, 2] = r_down * np.sin(theta_down)
        velocities[half:, 1] = -1.5  # Expanding downward
        
        # Rotation
        omega = 0.3
        velocities[:, 0] += -omega * positions[:, 2]
        velocities[:, 2] += omega * positions[:, 0]
        
    elif distribution == "fibonacci":
        # Fibonacci spiral (golden ratio spiral)
        golden_ratio = (1 + np.sqrt(5)) / 2
        golden_angle = 2 * np.pi / (golden_ratio ** 2)
        
        for i in range(n):
            # Fibonacci spiral in spherical coords
            theta = i * golden_angle
            r = R * np.sqrt(i / n)
            
            # Map to 3D spiral
            y = (i / n - 0.5) * R * 2  # Vertical spread
            
            positions[i, 0] = r * np.cos(theta)
            positions[i, 1] = y
            positions[i, 2] = r * np.sin(theta)
            
            # Orbital velocity
            if r > 0.01:
                v = np.sqrt(G * i * 0.001 / r)
                velocities[i, 0] = -v * np.sin(theta)
                velocities[i, 2] = v * np.cos(theta)
        
        velocities += np.random.normal(0, 0.2, (n, 3))
        
    elif distribution == "triple":
        # Three galaxies in triangular formation
        third = n // 3
        scale_length = R * 0.20
        softening = R * 0.02
        
        # Galaxy centers form equilateral triangle
        centers = np.array([
            [R * 0.6 * np.cos(0), 0, R * 0.6 * np.sin(0)],
            [R * 0.6 * np.cos(2*np.pi/3), 0, R * 0.6 * np.sin(2*np.pi/3)],
            [R * 0.6 * np.cos(4*np.pi/3), 0, R * 0.6 * np.sin(4*np.pi/3)],
        ])
        
        for gal_idx in range(3):
            start = gal_idx * third
            end = start + third if gal_idx < 2 else n
            gal_n = end - start
            
            # Radial distribution
            r = np.random.exponential(scale_length, gal_n)
            r = r * (1 - np.exp(-(R * 0.35) / (r + 0.01)))
            theta = np.random.uniform(0, 2 * np.pi, gal_n)
            z = np.random.normal(0, R * 0.01, gal_n)
            
            # Position relative to galaxy center
            positions[start:end, 0] = r * np.cos(theta) + centers[gal_idx, 0]
            positions[start:end, 1] = z + centers[gal_idx, 1]
            positions[start:end, 2] = r * np.sin(theta) + centers[gal_idx, 2]
            
            # Orbital velocity within galaxy
            masses_gal = masses[start:end]
            orbital_speed = compute_rotation_curve(r, masses_gal, G, softening)
            
            velocities[start:end, 0] = -orbital_speed * np.sin(theta)
            velocities[start:end, 2] = orbital_speed * np.cos(theta)
            
            # Add velocity dispersion
            sigma = orbital_speed * 0.10
            velocities[start:end, 0] += np.random.normal(0, sigma, gal_n)
            velocities[start:end, 1] += np.random.normal(0, sigma * 0.25, gal_n)
            velocities[start:end, 2] += np.random.normal(0, sigma, gal_n)
            
            # Bulk motion - galaxies orbit common center
            angle = gal_idx * 2 * np.pi / 3
            omega_bulk = 0.15
            velocities[start:end, 0] += -omega_bulk * centers[gal_idx, 2]
            velocities[start:end, 2] += omega_bulk * centers[gal_idx, 0]
        
    elif distribution == "rosette":
        # Flower-like orbital rosette pattern
        num_petals = 5
        petal_size = n // num_petals
        
        for petal in range(num_petals):
            start = petal * petal_size
            end = start + petal_size if petal < num_petals - 1 else n
            petal_n = end - start
            
            angle = petal * 2 * np.pi / num_petals
            
            # Each petal is an elliptical orbit
            r = np.random.exponential(R * 0.25, petal_n)
            theta = np.random.uniform(0, 2 * np.pi, petal_n)
            
            # Rotate each petal
            x_local = r * np.cos(theta)
            z_local = r * np.sin(theta) * 0.3  # Elliptical
            
            positions[start:end, 0] = x_local * np.cos(angle) - z_local * np.sin(angle)
            positions[start:end, 1] = np.random.normal(0, R * 0.02, petal_n)
            positions[start:end, 2] = x_local * np.sin(angle) + z_local * np.cos(angle)
            
            # Orbital motion
            omega = 0.5
            velocities[start:end, 0] = -omega * positions[start:end, 2]
            velocities[start:end, 2] = omega * positions[start:end, 0]
        
        velocities += np.random.normal(0, 0.1, (n, 3))
        
    elif distribution == "dyson":
        # Dyson sphere - particles orbit in spherical shell
        # (like a megastructure surrounding a star)
        
        # Spherical shell
        phi = np.random.uniform(0, 2 * np.pi, n)
        cos_theta = np.random.uniform(-1, 1, n)
        sin_theta = np.sqrt(1 - cos_theta**2)
        
        # Shell radius with some thickness
        r = R * 0.7 + np.random.normal(0, R * 0.03, n)
        
        positions[:, 0] = r * sin_theta * np.cos(phi)
        positions[:, 1] = r * cos_theta
        positions[:, 2] = r * sin_theta * np.sin(phi)
        
        # Central star mass
        central_mass = 5000.0
        
        # Orbital velocity for stable orbit
        v_orbital = np.sqrt(G * central_mass / r)
        
        # Tangent direction (perpendicular to radial)
        # Use cross product with random vector to get tangent
        radial_unit = positions / (r[:, np.newaxis] + 1e-10)
        
        # Tangent in horizontal plane
        velocities[:, 0] = -v_orbital * sin_theta * np.sin(phi)
        velocities[:, 1] = np.zeros(n)  # No vertical component
        velocities[:, 2] = v_orbital * sin_theta * np.cos(phi)
        
        # Small random motion
        velocities[:, 0] += np.random.normal(0, v_orbital * 0.05, n)
        velocities[:, 1] += np.random.normal(0, v_orbital * 0.05, n)
        velocities[:, 2] += np.random.normal(0, v_orbital * 0.05, n)
        
        masses[:] = 0.1  # Light structures
        
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
    "estimated_time": "~1 hour",
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
    "estimated_time": "~1 hour",
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
    "estimated_time": "~30 minutes",
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
    "estimated_time": "~30 minutes",
}

# -----------------------------------------------------------------------------
# CINEMATIC 4K PRESETS (Ultra quality, 60fps, high accuracy, long renders)
# For 4K 60fps footage - slow dt to prevent sped-up feel, high substeps
# -----------------------------------------------------------------------------

PRESETS["4k_galaxy_500k"] = {
    "name": "4K Galaxy 500K",
    "description": "500K body galaxy, 4K 60fps quality, high accuracy",
    "category": "CINEMATIC_4K",
    "num_bodies": 500_000,
    "theta": 0.5,  # High accuracy
    "G": 0.08,
    "softening": 1.5,
    "damping": 1.0,
    "spawn_radius": 600.0,
    "distribution": "galaxy",
    "total_frames": 3600,  # 1 minute at 60fps
    "dt_per_frame": 0.05,  # Slow motion - won't feel sped up
    "substeps": 5,  # Smooth physics
    "target_fps": 60,
    "estimated_time": "~5 hours",
}

PRESETS["4k_galaxy_1m"] = {
    "name": "4K Galaxy 1M",
    "description": "1 million body galaxy, ultra cinematic",
    "category": "CINEMATIC_4K",
    "num_bodies": 1_000_000,
    "theta": 0.5,
    "G": 0.07,
    "softening": 1.5,
    "damping": 1.0,
    "spawn_radius": 800.0,
    "distribution": "galaxy",
    "total_frames": 3600,
    "dt_per_frame": 0.05,
    "substeps": 5,
    "target_fps": 60,
    "estimated_time": "~11 hours",
}

PRESETS["4k_collision_500k"] = {
    "name": "4K Collision 500K",
    "description": "Two galaxies colliding, 4K 60fps, high accuracy",
    "category": "CINEMATIC_4K",
    "num_bodies": 500_000,
    "theta": 0.5,
    "G": 0.1,
    "softening": 1.5,
    "damping": 1.0,
    "spawn_radius": 700.0,
    "distribution": "collision",
    "total_frames": 6000,  # 100 seconds at 60fps
    "dt_per_frame": 0.06,
    "substeps": 5,
    "target_fps": 60,
    "estimated_time": "~9 hours",
}

PRESETS["4k_collision_1m"] = {
    "name": "4K Collision 1M",
    "description": "Epic 1M body collision, production quality",
    "category": "CINEMATIC_4K",
    "num_bodies": 1_000_000,
    "theta": 0.5,
    "G": 0.08,
    "softening": 1.5,
    "damping": 1.0,
    "spawn_radius": 900.0,
    "distribution": "collision",
    "total_frames": 6000,
    "dt_per_frame": 0.06,
    "substeps": 5,
    "target_fps": 60,
    "estimated_time": "~18 hours",
}

PRESETS["4k_spiral_500k"] = {
    "name": "4K Spiral 500K",
    "description": "Multi-arm spiral galaxy, 4K 60fps",
    "category": "CINEMATIC_4K",
    "num_bodies": 500_000,
    "theta": 0.5,
    "G": 0.06,
    "softening": 1.5,
    "damping": 1.0,
    "spawn_radius": 650.0,
    "distribution": "spiral",
    "total_frames": 3600,
    "dt_per_frame": 0.05,
    "substeps": 5,
    "target_fps": 60,
    "estimated_time": "~5 hours",
}

PRESETS["4k_spiral_1m"] = {
    "name": "4K Spiral 1M",
    "description": "Stunning 1M body spiral, ultra smooth",
    "category": "CINEMATIC_4K",
    "num_bodies": 1_000_000,
    "theta": 0.5,
    "G": 0.05,
    "softening": 1.5,
    "damping": 1.0,
    "spawn_radius": 850.0,
    "distribution": "spiral",
    "total_frames": 3600,
    "dt_per_frame": 0.05,
    "substeps": 5,
    "target_fps": 60,
    "estimated_time": "~11 hours",
}

PRESETS["4k_cluster_300k"] = {
    "name": "4K Globular Cluster",
    "description": "Dense star cluster, ultra accurate physics",
    "category": "CINEMATIC_4K",
    "num_bodies": 300_000,
    "theta": 0.4,  # Very accurate
    "G": 0.05,
    "softening": 1.0,
    "damping": 1.0,
    "spawn_radius": 300.0,
    "distribution": "cluster",
    "total_frames": 3600,
    "dt_per_frame": 0.04,
    "substeps": 6,  # Very smooth
    "target_fps": 60,
    "estimated_time": "~6 hours",
}

PRESETS["4k_ring_400k"] = {
    "name": "4K Saturn Rings",
    "description": "Beautiful ring system, cinematic quality",
    "category": "CINEMATIC_4K",
    "num_bodies": 400_000,
    "theta": 0.5,
    "G": 0.06,
    "softening": 1.0,
    "damping": 1.0,
    "spawn_radius": 400.0,
    "distribution": "ring",
    "total_frames": 3600,
    "dt_per_frame": 0.05,
    "substeps": 5,
    "target_fps": 60,
    "estimated_time": "~4 hours",
}

PRESETS["4k_binary_300k"] = {
    "name": "4K Binary System",
    "description": "Binary stars with disks, ultra smooth",
    "category": "CINEMATIC_4K",
    "num_bodies": 300_000,
    "theta": 0.5,
    "G": 0.12,
    "softening": 1.0,
    "damping": 1.0,
    "spawn_radius": 400.0,
    "distribution": "binary",
    "total_frames": 3600,
    "dt_per_frame": 0.05,
    "substeps": 5,
    "target_fps": 60,
    "estimated_time": "~3 hours",
}

# Long-form cinematic (2+ minutes at 60fps)

PRESETS["4k_galaxy_long"] = {
    "name": "4K Galaxy Long",
    "description": "Extended 2-minute galaxy evolution at 60fps",
    "category": "CINEMATIC_4K",
    "num_bodies": 500_000,
    "theta": 0.55,
    "G": 0.07,
    "softening": 1.5,
    "damping": 1.0,
    "spawn_radius": 650.0,
    "distribution": "galaxy",
    "total_frames": 7200,  # 2 minutes at 60fps
    "dt_per_frame": 0.05,
    "substeps": 4,
    "target_fps": 60,
    "estimated_time": "~7 hours",
}

PRESETS["4k_collision_epic"] = {
    "name": "4K Collision Epic",
    "description": "3-minute collision drama at 60fps",
    "category": "CINEMATIC_4K",
    "num_bodies": 600_000,
    "theta": 0.55,
    "G": 0.09,
    "softening": 1.5,
    "damping": 1.0,
    "spawn_radius": 800.0,
    "distribution": "collision",
    "total_frames": 10800,  # 3 minutes at 60fps
    "dt_per_frame": 0.06,
    "substeps": 4,
    "target_fps": 60,
    "estimated_time": "~12 hours",
}

PRESETS["4k_vortex_artistic"] = {
    "name": "4K Cosmic Vortex",
    "description": "Artistic swirling vortex, high frame count",
    "category": "CINEMATIC_4K",
    "num_bodies": 400_000,
    "theta": 0.5,
    "G": 0.06,
    "softening": 1.5,
    "damping": 0.998,
    "spawn_radius": 500.0,
    "distribution": "vortex",
    "total_frames": 6000,
    "dt_per_frame": 0.06,
    "substeps": 5,
    "target_fps": 60,
    "estimated_time": "~7 hours",
}

PRESETS["4k_supernova_burst"] = {
    "name": "4K Supernova",
    "description": "Explosive supernova at 60fps, high detail",
    "category": "CINEMATIC_4K",
    "num_bodies": 350_000,
    "theta": 0.5,
    "G": 0.04,
    "softening": 1.0,
    "damping": 0.995,
    "spawn_radius": 150.0,
    "distribution": "explosion",
    "total_frames": 3600,
    "dt_per_frame": 0.05,
    "substeps": 5,
    "target_fps": 60,
    "estimated_time": "~3 hours",
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
    "estimated_time": "~25 seconds",
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
    "estimated_time": "~25 seconds",
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
    "estimated_time": "~10 seconds",
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
    "estimated_time": "~10 seconds",
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
    "estimated_time": "~50 minutes",
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
    "estimated_time": "~35 minutes",
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
    "estimated_time": "~11 minutes",
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
    "estimated_time": "~3 minutes",
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
    "estimated_time": "~3 minutes",
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
    "estimated_time": "~6 minutes",
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
    "estimated_time": "~14 minutes",
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
    "estimated_time": "~3 minutes",
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
    "estimated_time": "~8 minutes",
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
    "estimated_time": "~10 minutes",
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
    "estimated_time": "~5 minutes",
}

PRESETS["cosmic_web"] = {
    "name": "Cosmic Web",
    "description": "Large-scale structure of the universe (needs millions)",
    "category": "ARTISTIC",
    "num_bodies": 500_000,
    "theta": 0.95,
    "G": 0.02,
    "softening": 5.0,
    "damping": 1.0,
    "spawn_radius": 1200.0,
    "distribution": "filament",
    "total_frames": 800,
    "dt_per_frame": 0.3,
    "substeps": 1,
    "target_fps": 24,
    "estimated_time": "~5 minutes",
}

PRESETS["dna_helix"] = {
    "name": "DNA Double Helix",
    "description": "Mesmerizing double helix structure",
    "category": "ARTISTIC",
    "num_bodies": 150_000,
    "theta": 0.9,
    "G": 0.05,
    "softening": 2.0,
    "damping": 1.0,
    "spawn_radius": 400.0,
    "distribution": "double_helix",
    "total_frames": 1200,
    "dt_per_frame": 0.1,
    "substeps": 2,
    "target_fps": 24,
    "estimated_time": "~4 minutes",
}

PRESETS["black_hole"] = {
    "name": "Black Hole Accretion",
    "description": "Accretion disk with brilliant jets",
    "category": "ARTISTIC",
    "num_bodies": 200_000,
    "theta": 0.85,
    "G": 0.3,
    "softening": 1.5,
    "damping": 1.0,
    "spawn_radius": 500.0,
    "distribution": "accretion_disk",
    "total_frames": 1500,
    "dt_per_frame": 0.08,
    "substeps": 3,
    "target_fps": 30,
    "estimated_time": "~6 minutes",
}

PRESETS["cosmic_donut"] = {
    "name": "Cosmic Torus",
    "description": "Beautiful donut-shaped structure",
    "category": "ARTISTIC",
    "num_bodies": 180_000,
    "theta": 0.88,
    "G": 0.08,
    "softening": 2.0,
    "damping": 1.0,
    "spawn_radius": 450.0,
    "distribution": "torus",
    "total_frames": 1200,
    "dt_per_frame": 0.12,
    "substeps": 2,
    "target_fps": 24,
    "estimated_time": "~5 minutes",
}

PRESETS["stellar_hourglass"] = {
    "name": "Stellar Hourglass",
    "description": "Binary star hourglass nebula",
    "category": "ARTISTIC",
    "num_bodies": 150_000,
    "theta": 0.9,
    "G": 0.1,
    "softening": 2.5,
    "damping": 1.0,
    "spawn_radius": 500.0,
    "distribution": "hourglass",
    "total_frames": 1000,
    "dt_per_frame": 0.15,
    "substeps": 2,
    "target_fps": 24,
    "estimated_time": "~4 minutes",
}

PRESETS["golden_spiral"] = {
    "name": "Fibonacci Spiral",
    "description": "Nature's golden ratio in space",
    "category": "ARTISTIC",
    "num_bodies": 120_000,
    "theta": 0.92,
    "G": 0.06,
    "softening": 2.0,
    "damping": 1.0,
    "spawn_radius": 450.0,
    "distribution": "fibonacci",
    "total_frames": 1200,
    "dt_per_frame": 0.12,
    "substeps": 2,
    "target_fps": 24,
    "estimated_time": "~3 minutes",
}

PRESETS["galactic_rosette"] = {
    "name": "Galactic Rosette",
    "description": "Flower-like orbital pattern",
    "category": "ARTISTIC",
    "num_bodies": 200_000,
    "theta": 0.88,
    "G": 0.1,
    "softening": 2.0,
    "damping": 1.0,
    "spawn_radius": 500.0,
    "distribution": "rosette",
    "total_frames": 1500,
    "dt_per_frame": 0.1,
    "substeps": 2,
    "target_fps": 24,
    "estimated_time": "~6 minutes",
}

PRESETS["dyson_sphere"] = {
    "name": "Dyson Sphere",
    "description": "Megastructure surrounding a star",
    "category": "ARTISTIC",
    "num_bodies": 250_000,
    "theta": 0.85,
    "G": 0.2,
    "softening": 1.5,
    "damping": 1.0,
    "spawn_radius": 600.0,
    "distribution": "dyson",
    "total_frames": 1500,
    "dt_per_frame": 0.08,
    "substeps": 3,
    "target_fps": 30,
    "estimated_time": "~8 minutes",
}

# -----------------------------------------------------------------------------
# MEGA PRESETS (Very large, long renders)
# -----------------------------------------------------------------------------

PRESETS["triple_collision"] = {
    "name": "Triple Collision",
    "description": "Three galaxies colliding chaotically",
    "category": "MEGA",
    "num_bodies": 300_000,
    "theta": 0.82,
    "G": 0.12,
    "softening": 2.5,
    "damping": 1.0,
    "spawn_radius": 800.0,
    "distribution": "triple",
    "total_frames": 2000,
    "dt_per_frame": 0.15,
    "substeps": 3,
    "target_fps": 24,
    "estimated_time": "~14 minutes",
}

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
    "estimated_time": "~40 minutes",
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
    "estimated_time": "~1 hour",
}

# -----------------------------------------------------------------------------
# EXTREME PRESETS (5M+ bodies, long offline renders)
# -----------------------------------------------------------------------------

PRESETS["extreme_5m_galaxy"] = {
    "name": "5 Million Star Galaxy",
    "description": "Massive galaxy with 5M bodies, approximate physics",
    "category": "EXTREME",
    "num_bodies": 5_000_000,
    "theta": 1.2,  # Very approximate for speed
    "G": 0.08,
    "softening": 5.0,
    "damping": 1.0,
    "spawn_radius": 1200.0,
    "distribution": "galaxy",
    "total_frames": 500,
    "dt_per_frame": 0.2,
    "substeps": 1,
    "target_fps": 20,
    "estimated_time": "~17 minutes",
}

PRESETS["extreme_5m_collision"] = {
    "name": "5 Million Collision",
    "description": "Epic collision with 5M bodies",
    "category": "EXTREME",
    "num_bodies": 5_000_000,
    "theta": 1.2,
    "G": 0.1,
    "softening": 5.0,
    "damping": 1.0,
    "spawn_radius": 1500.0,
    "distribution": "collision",
    "total_frames": 500,
    "dt_per_frame": 0.2,
    "substeps": 1,
    "target_fps": 20,
    "estimated_time": "~17 minutes",
}

PRESETS["extreme_5m_spiral"] = {
    "name": "5 Million Spiral",
    "description": "Gigantic spiral galaxy with 5M stars",
    "category": "EXTREME",
    "num_bodies": 5_000_000,
    "theta": 1.2,
    "G": 0.06,
    "softening": 5.0,
    "damping": 1.0,
    "spawn_radius": 1400.0,
    "distribution": "spiral",
    "total_frames": 500,
    "dt_per_frame": 0.2,
    "substeps": 1,
    "target_fps": 20,
    "estimated_time": "~17 minutes",
}

PRESETS["extreme_10m_galaxy"] = {
    "name": "10 Million Star Galaxy",
    "description": "Ultra-massive galaxy with 10M bodies",
    "category": "EXTREME",
    "num_bodies": 10_000_000,
    "theta": 1.3,  # More approximate
    "G": 0.06,
    "softening": 6.0,
    "damping": 1.0,
    "spawn_radius": 1600.0,
    "distribution": "galaxy",
    "total_frames": 500,
    "dt_per_frame": 0.25,
    "substeps": 1,
    "target_fps": 20,
    "estimated_time": "~30 minutes",
}

PRESETS["extreme_10m_collision"] = {
    "name": "10 Million Collision",
    "description": "Massive collision with 10M bodies",
    "category": "EXTREME",
    "num_bodies": 10_000_000,
    "theta": 1.3,
    "G": 0.08,
    "softening": 6.0,
    "damping": 1.0,
    "spawn_radius": 2000.0,
    "distribution": "collision",
    "total_frames": 500,
    "dt_per_frame": 0.25,
    "substeps": 1,
    "target_fps": 20,
    "estimated_time": "~30 minutes",
}

PRESETS["extreme_20m_galaxy"] = {
    "name": "20 Million Star Galaxy",
    "description": "Hyper-massive galaxy with 20M bodies",
    "category": "EXTREME",
    "num_bodies": 20_000_000,
    "theta": 1.4,  # Very coarse
    "G": 0.05,
    "softening": 8.0,
    "damping": 1.0,
    "spawn_radius": 2000.0,
    "distribution": "galaxy",
    "total_frames": 500,
    "dt_per_frame": 0.3,
    "substeps": 1,
    "target_fps": 20,
    "estimated_time": "~1 hour",
}

PRESETS["extreme_20m_spiral"] = {
    "name": "20 Million Spiral",
    "description": "Mega spiral galaxy with 20M stars",
    "category": "EXTREME",
    "num_bodies": 20_000_000,
    "theta": 1.4,
    "G": 0.04,
    "softening": 8.0,
    "damping": 1.0,
    "spawn_radius": 2200.0,
    "distribution": "spiral",
    "total_frames": 500,
    "dt_per_frame": 0.3,
    "substeps": 1,
    "target_fps": 20,
    "estimated_time": "~1 hour",
}

PRESETS["extreme_50m_galaxy"] = {
    "name": "50 Million Star Galaxy",
    "description": "Insane 50M body galaxy - multi-day render",
    "category": "EXTREME",
    "num_bodies": 50_000_000,
    "theta": 1.5,  # Maximum approximation
    "G": 0.04,
    "softening": 10.0,
    "damping": 1.0,
    "spawn_radius": 3000.0,
    "distribution": "galaxy",
    "total_frames": 500,
    "dt_per_frame": 0.35,
    "substeps": 1,
    "target_fps": 20,
    "estimated_time": "~2 hours",
}

PRESETS["extreme_50m_collision"] = {
    "name": "50 Million Collision",
    "description": "Ultimate collision with 50M bodies",
    "category": "EXTREME",
    "num_bodies": 50_000_000,
    "theta": 1.5,
    "G": 0.05,
    "softening": 10.0,
    "damping": 1.0,
    "spawn_radius": 3500.0,
    "distribution": "collision",
    "total_frames": 500,
    "dt_per_frame": 0.35,
    "substeps": 1,
    "target_fps": 20,
    "estimated_time": "~2 hours",
}

PRESETS["extreme_50m_web"] = {
    "name": "50 Million Cosmic Web",
    "description": "Ultimate cosmic web - CMB-like large scale structure",
    "category": "EXTREME",
    "num_bodies": 50_000_000,
    "theta": 1.5,
    "G": 0.01,
    "softening": 15.0,
    "damping": 1.0,
    "spawn_radius": 5000.0,
    "distribution": "filament",
    "total_frames": 500,
    "dt_per_frame": 0.4,
    "substeps": 1,
    "target_fps": 20,
    "estimated_time": "~2 hours",
}

PRESETS["extreme_20m_web"] = {
    "name": "20 Million Cosmic Web",
    "description": "Massive cosmic web structure",
    "category": "EXTREME",
    "num_bodies": 20_000_000,
    "theta": 1.4,
    "G": 0.015,
    "softening": 12.0,
    "damping": 1.0,
    "spawn_radius": 4000.0,
    "distribution": "filament",
    "total_frames": 500,
    "dt_per_frame": 0.4,
    "substeps": 1,
    "target_fps": 20,
    "estimated_time": "~1 hour",
}

PRESETS["extreme_10m_web"] = {
    "name": "10 Million Cosmic Web",
    "description": "Large cosmic web with filaments and voids",
    "category": "EXTREME",
    "num_bodies": 10_000_000,
    "theta": 1.3,
    "G": 0.02,
    "softening": 10.0,
    "damping": 1.0,
    "spawn_radius": 3000.0,
    "distribution": "filament",
    "total_frames": 500,
    "dt_per_frame": 0.35,
    "substeps": 1,
    "target_fps": 20,
    "estimated_time": "~30 minutes",
}

PRESETS["extreme_5m_web"] = {
    "name": "5 Million Cosmic Web",
    "description": "Cosmic web with clear filamentary structure",
    "category": "EXTREME",
    "num_bodies": 5_000_000,
    "theta": 1.2,
    "G": 0.025,
    "softening": 8.0,
    "damping": 1.0,
    "spawn_radius": 2500.0,
    "distribution": "filament",
    "total_frames": 500,
    "dt_per_frame": 0.35,
    "substeps": 1,
    "target_fps": 20,
    "estimated_time": "~17 minutes",
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
    "estimated_time": "~3 seconds",
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
    "estimated_time": "~5 seconds",
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
    "estimated_time": "~5 seconds",
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_preset_list() -> List[Tuple[str, dict]]:
    """Get list of all presets sorted by category."""
    category_order = ["TINY", "FAST", "CINEMATIC", "CINEMATIC_4K", "ARTISTIC", "SCIENTIFIC", "CHAOS", "MEGA", "EXTREME"]
    
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

