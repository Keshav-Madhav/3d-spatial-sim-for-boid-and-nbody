"""
Ultra-optimized N-body gravitational simulation using Barnes-Hut algorithm.
Handles 1,000,000+ bodies at interactive framerates.

Key optimizations:
- Barnes-Hut octree reduces O(n²) to O(n log n)
- Numba JIT compilation with parallel processing
- Flattened array-based octree (no Python objects)
- VBO point sprite rendering (1 vertex per body)
- Adaptive time stepping
"""

import math
import numpy as np
from numba import njit, prange
from OpenGL.GL import *
from OpenGL.arrays import vbo

from config import nbody as config


# ============================================================================
# BARNES-HUT OCTREE - FLATTENED ARRAY IMPLEMENTATION
# ============================================================================

# Octree node structure (stored in flat arrays):
# - center_x, center_y, center_z: node center
# - half_size: half the width of the node
# - mass: total mass in this node
# - com_x, com_y, com_z: center of mass
# - children[8]: indices to child nodes (-1 if none)
# - body_index: index of single body in leaf (-1 if internal or empty)
# - is_leaf: whether this is a leaf node

MAX_TREE_NODES = 8_000_000  # Pre-allocate for large simulations


@njit(cache=True)
def get_octant(px: float, py: float, pz: float, 
               cx: float, cy: float, cz: float) -> int:
    """Determine which octant a point falls into relative to center."""
    octant = 0
    if px >= cx:
        octant |= 1
    if py >= cy:
        octant |= 2
    if pz >= cz:
        octant |= 4
    return octant


@njit(cache=True)
def get_octant_center(octant: int, cx: float, cy: float, cz: float, 
                      half_size: float) -> tuple:
    """Get the center of a child octant."""
    quarter = half_size * 0.5
    new_cx = cx + quarter if (octant & 1) else cx - quarter
    new_cy = cy + quarter if (octant & 2) else cy - quarter
    new_cz = cz + quarter if (octant & 4) else cz - quarter
    return new_cx, new_cy, new_cz


@njit(cache=True)
def build_octree(
    positions: np.ndarray,
    masses: np.ndarray,
    num_bodies: int,
    bounds: float,
    # Output arrays (pre-allocated)
    node_centers: np.ndarray,      # (max_nodes, 3)
    node_half_sizes: np.ndarray,   # (max_nodes,)
    node_masses: np.ndarray,       # (max_nodes,)
    node_com: np.ndarray,          # (max_nodes, 3)
    node_children: np.ndarray,     # (max_nodes, 8)
    node_body_idx: np.ndarray,     # (max_nodes,)
    node_is_leaf: np.ndarray,      # (max_nodes,)
) -> int:
    """
    Build Barnes-Hut octree from body positions.
    Returns number of nodes created.
    """
    # Initialize root node
    node_centers[0, 0] = 0.0
    node_centers[0, 1] = 0.0
    node_centers[0, 2] = 0.0
    node_half_sizes[0] = bounds
    node_masses[0] = 0.0
    node_com[0, 0] = 0.0
    node_com[0, 1] = 0.0
    node_com[0, 2] = 0.0
    node_body_idx[0] = -1
    node_is_leaf[0] = True
    for c in range(8):
        node_children[0, c] = -1
    
    num_nodes = 1
    
    # Insert each body into the tree
    for i in range(num_bodies):
        px, py, pz = positions[i, 0], positions[i, 1], positions[i, 2]
        m = masses[i]
        
        # Start at root
        current = 0
        
        # Traverse tree to find insertion point
        while True:
            cx = node_centers[current, 0]
            cy = node_centers[current, 1]
            cz = node_centers[current, 2]
            hs = node_half_sizes[current]
            
            if node_is_leaf[current]:
                if node_body_idx[current] == -1:
                    # Empty leaf - insert body here
                    node_body_idx[current] = i
                    node_masses[current] = m
                    node_com[current, 0] = px
                    node_com[current, 1] = py
                    node_com[current, 2] = pz
                    break
                else:
                    # Occupied leaf - subdivide
                    old_body = node_body_idx[current]
                    old_px = positions[old_body, 0]
                    old_py = positions[old_body, 1]
                    old_pz = positions[old_body, 2]
                    old_m = masses[old_body]
                    
                    # Mark as internal node
                    node_is_leaf[current] = False
                    node_body_idx[current] = -1
                    
                    # Re-insert old body
                    octant = get_octant(old_px, old_py, old_pz, cx, cy, cz)
                    
                    if node_children[current, octant] == -1:
                        # Create child node
                        child_idx = num_nodes
                        num_nodes += 1
                        if num_nodes >= MAX_TREE_NODES:
                            break
                        
                        node_children[current, octant] = child_idx
                        new_cx, new_cy, new_cz = get_octant_center(octant, cx, cy, cz, hs)
                        node_centers[child_idx, 0] = new_cx
                        node_centers[child_idx, 1] = new_cy
                        node_centers[child_idx, 2] = new_cz
                        node_half_sizes[child_idx] = hs * 0.5
                        node_masses[child_idx] = old_m
                        node_com[child_idx, 0] = old_px
                        node_com[child_idx, 1] = old_py
                        node_com[child_idx, 2] = old_pz
                        node_body_idx[child_idx] = old_body
                        node_is_leaf[child_idx] = True
                        for c in range(8):
                            node_children[child_idx, c] = -1
                    
                    # Continue inserting new body (don't break, loop will handle it)
            else:
                # Internal node - update mass and COM
                total_mass = node_masses[current] + m
                if total_mass > 0:
                    node_com[current, 0] = (node_com[current, 0] * node_masses[current] + px * m) / total_mass
                    node_com[current, 1] = (node_com[current, 1] * node_masses[current] + py * m) / total_mass
                    node_com[current, 2] = (node_com[current, 2] * node_masses[current] + pz * m) / total_mass
                node_masses[current] = total_mass
                
                # Find appropriate child
                octant = get_octant(px, py, pz, cx, cy, cz)
                
                if node_children[current, octant] == -1:
                    # Create child node
                    child_idx = num_nodes
                    num_nodes += 1
                    if num_nodes >= MAX_TREE_NODES:
                        break
                    
                    node_children[current, octant] = child_idx
                    new_cx, new_cy, new_cz = get_octant_center(octant, cx, cy, cz, hs)
                    node_centers[child_idx, 0] = new_cx
                    node_centers[child_idx, 1] = new_cy
                    node_centers[child_idx, 2] = new_cz
                    node_half_sizes[child_idx] = hs * 0.5
                    node_masses[child_idx] = m
                    node_com[child_idx, 0] = px
                    node_com[child_idx, 1] = py
                    node_com[child_idx, 2] = pz
                    node_body_idx[child_idx] = i
                    node_is_leaf[child_idx] = True
                    for c in range(8):
                        node_children[child_idx, c] = -1
                    break
                else:
                    # Traverse to child
                    current = node_children[current, octant]
    
    return num_nodes


@njit(parallel=True, fastmath=True, cache=True)
def compute_forces_barnes_hut(
    positions: np.ndarray,
    masses: np.ndarray,
    accelerations: np.ndarray,
    node_centers: np.ndarray,
    node_half_sizes: np.ndarray,
    node_masses: np.ndarray,
    node_com: np.ndarray,
    node_children: np.ndarray,
    node_body_idx: np.ndarray,
    node_is_leaf: np.ndarray,
    num_nodes: int,
    num_bodies: int,
    theta: float,
    G: float,
    softening: float
):
    """
    Compute gravitational forces using Barnes-Hut tree traversal.
    Uses stack-based traversal to avoid recursion (Numba-friendly).
    """
    softening_sq = softening * softening
    
    for i in prange(num_bodies):
        px = positions[i, 0]
        py = positions[i, 1]
        pz = positions[i, 2]
        
        ax, ay, az = 0.0, 0.0, 0.0
        
        # Stack-based tree traversal (fixed-size stack)
        stack = np.zeros(64, dtype=np.int32)
        stack[0] = 0  # Start at root
        stack_ptr = 1
        
        while stack_ptr > 0:
            stack_ptr -= 1
            node = stack[stack_ptr]
            
            if node < 0 or node >= num_nodes:
                continue
            
            # Skip if this node contains only this body
            if node_is_leaf[node] and node_body_idx[node] == i:
                continue
            
            # Calculate distance to node's center of mass
            dx = node_com[node, 0] - px
            dy = node_com[node, 1] - py
            dz = node_com[node, 2] - pz
            dist_sq = dx * dx + dy * dy + dz * dz + softening_sq
            dist = math.sqrt(dist_sq)
            
            # Barnes-Hut criterion: s/d < theta
            node_size = node_half_sizes[node] * 2.0
            
            if node_is_leaf[node] or (node_size / dist < theta):
                # Use this node's center of mass as approximation
                if node_masses[node] > 0 and dist_sq > softening_sq:
                    # F = G * m1 * m2 / r^2, a = F / m1 = G * m2 / r^2
                    inv_dist3 = 1.0 / (dist * dist_sq)
                    force_mag = G * node_masses[node] * inv_dist3
                    
                    ax += dx * force_mag
                    ay += dy * force_mag
                    az += dz * force_mag
            else:
                # Node too close, need to examine children
                for c in range(8):
                    child = node_children[node, c]
                    if child >= 0 and stack_ptr < 64:
                        stack[stack_ptr] = child
                        stack_ptr += 1
        
        accelerations[i, 0] = ax
        accelerations[i, 1] = ay
        accelerations[i, 2] = az


@njit(parallel=True, fastmath=True, cache=True)
def update_positions_velocities(
    positions: np.ndarray,
    velocities: np.ndarray,
    accelerations: np.ndarray,
    damping: float,
    dt: float,
    num_bodies: int
):
    """Update positions and velocities using leapfrog integration. No boundaries."""
    for i in prange(num_bodies):
        # Update velocity
        velocities[i, 0] += accelerations[i, 0] * dt
        velocities[i, 1] += accelerations[i, 1] * dt
        velocities[i, 2] += accelerations[i, 2] * dt
        
        # Apply damping if configured (1.0 = no damping)
        velocities[i, 0] *= damping
        velocities[i, 1] *= damping
        velocities[i, 2] *= damping
        
        # Update position - no boundaries, bodies can escape
        positions[i, 0] += velocities[i, 0] * dt
        positions[i, 1] += velocities[i, 1] * dt
        positions[i, 2] += velocities[i, 2] * dt


@njit(fastmath=True, cache=True)
def compute_bounds(positions: np.ndarray, num_bodies: int) -> float:
    """Compute the bounding box extent of all particles."""
    max_extent = 0.0
    for i in range(num_bodies):
        for dim in range(3):
            ext = abs(positions[i, dim])
            if ext > max_extent:
                max_extent = ext
    return max_extent * 1.1 + 10.0  # Add margin


@njit(parallel=True, fastmath=True, cache=True)
def compute_colors_by_velocity(
    velocities: np.ndarray,
    colors: np.ndarray,
    num_bodies: int,
    max_speed: float
):
    """Color bodies based on velocity magnitude (bright blue-purple → red heat map).
    
    Color distribution:
    - 0-30%: Bright purple-blue → Blue → Light blue (slow, visible against black)
    - 30-55%: Light blue → Cyan → White (normal speed)
    - 55-90%: White (high speed - PRIMARY)
    - 90-95%: White → Yellow (very high speed)
    - 95-99%: Yellow → Orange (rare, extreme speed)
    - 99-100%: Orange → Red (extremely rare, maximum speed)
    """
    for i in prange(num_bodies):
        speed = math.sqrt(
            velocities[i, 0] ** 2 + 
            velocities[i, 1] ** 2 + 
            velocities[i, 2] ** 2
        )
        t = min(1.0, speed / max_speed)
        
        # Color gradient: bright purple-blue → blue → light blue → cyan → white → yellow → orange → red
        # Minimum brightness increased to ensure visibility against black background
        # More color variation in slow range for better differentiation
        
        if t < 0.55:
            if t < 0.15:
                # Bright purple-blue (0.4, 0.2, 0.8) → Blue (0.2, 0.4, 0.9)
                # Very slow bodies: bright enough to see, purple tint distinguishes them
                s = t / 0.15
                colors[i, 0] = 0.4 - 0.2 * s
                colors[i, 1] = 0.2 + 0.2 * s
                colors[i, 2] = 0.8 + 0.1 * s
            elif t < 0.30:
                # Blue (0.2, 0.4, 0.9) → Light blue (0.3, 0.5, 0.95)
                s = (t - 0.15) / 0.15
                colors[i, 0] = 0.2 + 0.1 * s
                colors[i, 1] = 0.4 + 0.1 * s
                colors[i, 2] = 0.9 + 0.05 * s
            else:
                # Light blue → Cyan → White
                s = (t - 0.30) / 0.25  # 0 to 1 over 0.30-0.55 range
                if s < 0.6:
                    # Light blue → Cyan
                    s2 = s / 0.6
                    colors[i, 0] = 0.3 - 0.1 * s2
                    colors[i, 1] = 0.5 + 0.3 * s2
                    colors[i, 2] = 0.95 + 0.05 * s2
                else:
                    # Cyan → White
                    s2 = (s - 0.6) / 0.4
                    colors[i, 0] = 0.2 + 0.8 * s2
                    colors[i, 1] = 0.8 + 0.2 * s2
                    colors[i, 2] = 1.0
        elif t < 0.90:
            # White (1.0, 1.0, 1.0) - PRIMARY RANGE
            colors[i, 0] = 1.0
            colors[i, 1] = 1.0
            colors[i, 2] = 1.0
        elif t < 0.95:
            # White (1.0, 1.0, 1.0) → Yellow (1.0, 0.95, 0.0)
            s = (t - 0.90) / 0.05
            colors[i, 0] = 1.0
            colors[i, 1] = 1.0 - 0.05 * s
            colors[i, 2] = 1.0 - 1.0 * s
        elif t < 0.99:
            # Yellow (1.0, 0.95, 0.0) → Orange (1.0, 0.5, 0.0) - RARE
            s = (t - 0.95) / 0.04
            colors[i, 0] = 1.0
            colors[i, 1] = 0.95 - 0.45 * s
            colors[i, 2] = 0.0
        else:
            # Orange (1.0, 0.5, 0.0) → Red (1.0, 0.0, 0.0) - EXTREMELY RARE!
            s = (t - 0.99) / 0.01
            colors[i, 0] = 1.0
            colors[i, 1] = 0.5 - 0.5 * s
            colors[i, 2] = 0.0


@njit(parallel=True, fastmath=True, cache=True)
def compute_visibility_points(
    positions: np.ndarray,
    cam_pos: np.ndarray,
    cam_forward: np.ndarray,
    cam_right: np.ndarray,
    cam_up: np.ndarray,
    tan_h: float,
    tan_v: float,
    far_dist: float,
    visible_mask: np.ndarray,
    num_bodies: int
):
    """Frustum culling for point rendering."""
    for i in prange(num_bodies):
        dx = positions[i, 0] - cam_pos[0]
        dy = positions[i, 1] - cam_pos[1]
        dz = positions[i, 2] - cam_pos[2]
        
        z = dx * cam_forward[0] + dy * cam_forward[1] + dz * cam_forward[2]
        
        if z < 0.1 or z > far_dist:
            visible_mask[i] = False
            continue
        
        x = dx * cam_right[0] + dy * cam_right[1] + dz * cam_right[2]
        y = dx * cam_up[0] + dy * cam_up[1] + dz * cam_up[2]
        
        half_width = z * tan_h * 1.2  # Slight margin
        half_height = z * tan_v * 1.2
        
        visible_mask[i] = abs(x) < half_width and abs(y) < half_height


# ============================================================================
# N-BODY SIMULATION CLASS
# ============================================================================

class NBodySimulation:
    """
    Ultra-high-performance N-body gravitational simulation.
    
    Automatically uses the best available backend:
    - CUDA (NVIDIA GPUs) - Full GPU acceleration
    - Metal/MPS (Apple Silicon) - Full GPU acceleration  
    - CPU (fallback) - Barnes-Hut with Numba parallel
    """
    
    def __init__(self, num_bodies: int = 1_000_000):
        self.num_bodies = num_bodies
        
        # Load config
        nbody_cfg = config.NBODY
        self.spawn_radius = float(nbody_cfg["spawn_radius"])
        self.G = float(nbody_cfg["G"])
        self.theta = float(nbody_cfg["theta"])
        self.softening = float(nbody_cfg["softening"])
        self.damping = float(nbody_cfg["damping"])
        self.point_size = float(nbody_cfg["point_size"])
        self.max_speed_color = float(nbody_cfg["max_speed_color"])
        
        # Dynamic bounds (computed each frame from particle positions)
        self.current_bounds = self.spawn_radius * 2
        
        # Initialize body arrays
        self._init_bodies(nbody_cfg)
        
        # Try to initialize GPU backend
        self._gpu_sim = None
        self._use_gpu = False
        self._init_gpu_backend()
        
        # Octree arrays (pre-allocated) - only needed for CPU fallback
        if not self._use_gpu:
            max_nodes = min(MAX_TREE_NODES, num_bodies * 4)
            self._node_centers = np.zeros((max_nodes, 3), dtype=np.float64)
            self._node_half_sizes = np.zeros(max_nodes, dtype=np.float64)
            self._node_masses = np.zeros(max_nodes, dtype=np.float64)
            self._node_com = np.zeros((max_nodes, 3), dtype=np.float64)
            self._node_children = np.full((max_nodes, 8), -1, dtype=np.int32)
            self._node_body_idx = np.full(max_nodes, -1, dtype=np.int32)
            self._node_is_leaf = np.ones(max_nodes, dtype=np.bool_)
        self._num_tree_nodes = 0
        
        # Rendering
        self._visible_mask = np.ones(num_bodies, dtype=np.bool_)
        self._visible_count = num_bodies
        self._cam_pos = np.zeros(3, dtype=np.float64)
        self._cam_forward = np.zeros(3, dtype=np.float64)
        self._cam_right = np.zeros(3, dtype=np.float64)
        self._cam_up = np.zeros(3, dtype=np.float64)
        
        # VBOs
        self._vbo_positions = None
        self._vbo_colors = None
        self._vbos_initialized = False
        
        # Fog/culling
        self.fog_end = float(config.CAMERA["far_clip"])
        
        # Warm up Numba (only for CPU)
        if not self._use_gpu:
            self._warmup_numba()
        
        print(f"[NBody] Initialized {num_bodies:,} bodies")
    
    def _init_gpu_backend(self):
        """Try to initialize GPU acceleration."""
        try:
            from .gpu_backend import get_backend, Backend, create_gpu_simulation
            
            backend, info = get_backend()
            
            if backend != Backend.CPU:
                self._gpu_sim = create_gpu_simulation(
                    self.positions,
                    self.velocities,
                    self.masses,
                    self.G,
                    self.softening,
                    self.damping,
                    theta=self.theta  # Pass theta for Barnes-Hut backends
                )
                if self._gpu_sim is not None:
                    self._use_gpu = True
                    self._backend = backend
                    print(f"[NBody] GPU acceleration enabled: {backend.value}")
                    if backend == Backend.METAL_BH:
                        print(f"[NBody] Using Metal Barnes-Hut (θ={self.theta}, UMA zero-copy)")
                    return
        except ImportError as e:
            print(f"[NBody] GPU backend not available: {e}")
        except Exception as e:
            print(f"[NBody] GPU init failed, using CPU: {e}")
        
        self._use_gpu = False
        self._backend = Backend.CPU if 'Backend' in dir() else None
        print("[NBody] Using CPU backend (Barnes-Hut + Numba)")
    
    def _init_bodies(self, cfg):
        """Initialize body positions, velocities, masses based on distribution."""
        distribution = cfg.get("distribution", "galaxy")
        
        if distribution == "galaxy":
            self._init_galaxy_distribution(cfg)
        elif distribution == "spiral":
            self._init_spiral_distribution(cfg)
        elif distribution == "sphere":
            self._init_sphere_distribution(cfg)
        elif distribution == "collision":
            self._init_collision_distribution(cfg)
        else:
            self._init_uniform_distribution(cfg)
    
    def _init_galaxy_distribution(self, cfg):
        """Initialize bodies in a rotating disk galaxy pattern."""
        n = self.num_bodies
        R = self.spawn_radius
        
        # Disk-shaped distribution
        r = np.random.exponential(R * 0.3, n)
        theta = np.random.uniform(0, 2 * np.pi, n)
        z = np.random.normal(0, R * 0.02, n)
        
        self.positions = np.zeros((n, 3), dtype=np.float64)
        self.positions[:, 0] = r * np.cos(theta)
        self.positions[:, 1] = z
        self.positions[:, 2] = r * np.sin(theta)
        
        # Keplerian-ish rotation
        self.velocities = np.zeros((n, 3), dtype=np.float64)
        orbital_speed = np.sqrt(self.G * n * 0.001 / (r + 1.0))
        self.velocities[:, 0] = -orbital_speed * np.sin(theta)
        self.velocities[:, 2] = orbital_speed * np.cos(theta)
        
        # Add some vertical velocity dispersion
        self.velocities[:, 1] = np.random.normal(0, orbital_speed * 0.1, n)
        
        # Masses - mostly uniform with some heavier particles
        self.masses = np.ones(n, dtype=np.float64)
        heavy_count = max(1, n // 1000)
        heavy_indices = np.random.choice(n, heavy_count, replace=False)
        self.masses[heavy_indices] = 100.0
        
        self.accelerations = np.zeros((n, 3), dtype=np.float64)
        self.colors = np.zeros((n, 3), dtype=np.float32)
    
    def _init_spiral_distribution(self, cfg):
        """
        Initialize a Milky Way-like spiral galaxy with:
        - Massive central black hole / bulge
        - Spiral arms with logarithmic spiral pattern
        - Disk of stars with proper orbital velocities
        """
        n = self.num_bodies
        R = self.spawn_radius
        
        self.positions = np.zeros((n, 3), dtype=np.float64)
        self.velocities = np.zeros((n, 3), dtype=np.float64)
        self.masses = np.ones(n, dtype=np.float64)
        
        # Reserve first particle as super-massive central object
        central_mass = n * 50.0  # Very massive center
        self.masses[0] = central_mass
        self.positions[0] = [0, 0, 0]
        self.velocities[0] = [0, 0, 0]
        
        # Bulge particles (inner 5% of bodies) - dense spherical core
        bulge_count = max(1, n // 20)
        bulge_r = np.random.exponential(R * 0.05, bulge_count)
        bulge_theta = np.random.uniform(0, 2 * np.pi, bulge_count)
        bulge_phi = np.arccos(np.random.uniform(-1, 1, bulge_count))
        
        self.positions[1:bulge_count+1, 0] = bulge_r * np.sin(bulge_phi) * np.cos(bulge_theta)
        self.positions[1:bulge_count+1, 1] = bulge_r * np.sin(bulge_phi) * np.sin(bulge_theta) * 0.3  # Flattened
        self.positions[1:bulge_count+1, 2] = bulge_r * np.cos(bulge_phi)
        
        # Bulge velocities - random but bound
        bulge_orbital = np.sqrt(self.G * central_mass / (bulge_r + 1.0)) * 0.5
        self.velocities[1:bulge_count+1, 0] = np.random.normal(0, bulge_orbital * 0.3, bulge_count)
        self.velocities[1:bulge_count+1, 1] = np.random.normal(0, bulge_orbital * 0.1, bulge_count)
        self.velocities[1:bulge_count+1, 2] = np.random.normal(0, bulge_orbital * 0.3, bulge_count)
        
        # Disk particles with spiral arms
        disk_start = bulge_count + 1
        disk_count = n - disk_start
        
        # Number of spiral arms
        num_arms = 4
        arm_spread = 0.3  # How spread out the arms are
        
        # Logarithmic spiral: r = a * e^(b * theta)
        # We'll use: theta = log(r/a) / b, then add arm offset
        
        # Radial distribution - exponential disk
        disk_r = np.random.exponential(R * 0.25, disk_count)
        disk_r = np.clip(disk_r, R * 0.02, R * 0.9)  # Keep within bounds
        
        # Base angle from logarithmic spiral
        spiral_tightness = 0.3  # Lower = tighter spiral
        base_theta = np.log(disk_r / (R * 0.05) + 1) / spiral_tightness
        
        # Assign to arms with some spread
        arm_assignment = np.random.randint(0, num_arms, disk_count)
        arm_offset = arm_assignment * (2 * np.pi / num_arms)
        
        # Add randomness to create arm width
        theta_scatter = np.random.normal(0, arm_spread, disk_count)
        
        disk_theta = base_theta + arm_offset + theta_scatter
        
        # Vertical distribution - thin disk with some thickness
        disk_z = np.random.normal(0, R * 0.01, disk_count) * (1 + disk_r / R)  # Flaring
        
        # Convert to Cartesian
        self.positions[disk_start:, 0] = disk_r * np.cos(disk_theta)
        self.positions[disk_start:, 1] = disk_z
        self.positions[disk_start:, 2] = disk_r * np.sin(disk_theta)
        
        # Orbital velocities - circular orbits around center
        # v = sqrt(G * M_enclosed / r), approximate M_enclosed
        enclosed_mass = central_mass + disk_r / R * n * 0.5  # Rough approximation
        orbital_speed = np.sqrt(self.G * enclosed_mass / (disk_r + 0.1))
        
        # Tangential velocity (perpendicular to radius)
        self.velocities[disk_start:, 0] = -orbital_speed * np.sin(disk_theta)
        self.velocities[disk_start:, 2] = orbital_speed * np.cos(disk_theta)
        
        # Small velocity dispersion
        self.velocities[disk_start:, 0] += np.random.normal(0, orbital_speed * 0.05, disk_count)
        self.velocities[disk_start:, 1] = np.random.normal(0, orbital_speed * 0.02, disk_count)
        self.velocities[disk_start:, 2] += np.random.normal(0, orbital_speed * 0.05, disk_count)
        
        self.accelerations = np.zeros((n, 3), dtype=np.float64)
        self.colors = np.zeros((n, 3), dtype=np.float32)
    
    def _init_sphere_distribution(self, cfg):
        """Initialize bodies in a uniform sphere."""
        n = self.num_bodies
        R = self.spawn_radius
        
        # Uniform sphere distribution
        phi = np.random.uniform(0, 2 * np.pi, n)
        cos_theta = np.random.uniform(-1, 1, n)
        sin_theta = np.sqrt(1 - cos_theta**2)
        r = R * 0.8 * np.cbrt(np.random.uniform(0, 1, n))
        
        self.positions = np.zeros((n, 3), dtype=np.float64)
        self.positions[:, 0] = r * sin_theta * np.cos(phi)
        self.positions[:, 1] = r * sin_theta * np.sin(phi)
        self.positions[:, 2] = r * cos_theta
        
        # Small random velocities
        self.velocities = np.random.normal(0, 0.5, (n, 3)).astype(np.float64)
        
        self.masses = np.ones(n, dtype=np.float64)
        self.accelerations = np.zeros((n, 3), dtype=np.float64)
        self.colors = np.zeros((n, 3), dtype=np.float32)
    
    def _init_collision_distribution(self, cfg):
        """Initialize two galaxies on collision course."""
        n = self.num_bodies
        half = n // 2
        R = self.spawn_radius
        
        self.positions = np.zeros((n, 3), dtype=np.float64)
        self.velocities = np.zeros((n, 3), dtype=np.float64)
        
        # Galaxy 1
        r1 = np.random.exponential(R * 0.2, half)
        theta1 = np.random.uniform(0, 2 * np.pi, half)
        self.positions[:half, 0] = r1 * np.cos(theta1) - R * 0.4
        self.positions[:half, 1] = np.random.normal(0, R * 0.02, half)
        self.positions[:half, 2] = r1 * np.sin(theta1)
        
        orbital_speed1 = np.sqrt(self.G * half * 0.001 / (r1 + 1.0))
        self.velocities[:half, 0] = -orbital_speed1 * np.sin(theta1) + 2.0
        self.velocities[:half, 2] = orbital_speed1 * np.cos(theta1)
        
        # Galaxy 2
        r2 = np.random.exponential(R * 0.2, n - half)
        theta2 = np.random.uniform(0, 2 * np.pi, n - half)
        self.positions[half:, 0] = r2 * np.cos(theta2) + R * 0.4
        self.positions[half:, 1] = np.random.normal(0, R * 0.02, n - half)
        self.positions[half:, 2] = r2 * np.sin(theta2)
        
        orbital_speed2 = np.sqrt(self.G * (n - half) * 0.001 / (r2 + 1.0))
        self.velocities[half:, 0] = -orbital_speed2 * np.sin(theta2) - 2.0
        self.velocities[half:, 2] = orbital_speed2 * np.cos(theta2)
        
        self.masses = np.ones(n, dtype=np.float64)
        self.accelerations = np.zeros((n, 3), dtype=np.float64)
        self.colors = np.zeros((n, 3), dtype=np.float32)
    
    def _init_uniform_distribution(self, cfg):
        """Initialize bodies uniformly in a cube."""
        n = self.num_bodies
        R = self.spawn_radius
        
        self.positions = ((np.random.rand(n, 3) - 0.5) * 2 * R * 0.8).astype(np.float64)
        self.velocities = np.random.normal(0, 1.0, (n, 3)).astype(np.float64)
        self.masses = np.ones(n, dtype=np.float64)
        self.accelerations = np.zeros((n, 3), dtype=np.float64)
        self.colors = np.zeros((n, 3), dtype=np.float32)
    
    def _warmup_numba(self):
        """Pre-compile Numba functions with small arrays."""
        n = 100
        pos = np.random.rand(n, 3).astype(np.float64) * 10
        vel = np.random.rand(n, 3).astype(np.float64)
        mass = np.ones(n, dtype=np.float64)
        acc = np.zeros((n, 3), dtype=np.float64)
        colors = np.zeros((n, 3), dtype=np.float32)
        
        max_nodes = 1000
        centers = np.zeros((max_nodes, 3), dtype=np.float64)
        half_sizes = np.zeros(max_nodes, dtype=np.float64)
        masses_tree = np.zeros(max_nodes, dtype=np.float64)
        com = np.zeros((max_nodes, 3), dtype=np.float64)
        children = np.full((max_nodes, 8), -1, dtype=np.int32)
        body_idx = np.full(max_nodes, -1, dtype=np.int32)
        is_leaf = np.ones(max_nodes, dtype=np.bool_)
        
        num_nodes = build_octree(pos, mass, n, 50.0, centers, half_sizes, 
                                  masses_tree, com, children, body_idx, is_leaf)
        
        compute_forces_barnes_hut(pos, mass, acc, centers, half_sizes, masses_tree,
                                   com, children, body_idx, is_leaf, num_nodes, n,
                                   0.5, 1.0, 0.1)
        
        update_positions_velocities(pos, vel, acc, 1.0, 0.01, n)
        compute_colors_by_velocity(vel, colors, n, 10.0)
        compute_bounds(pos, n)  # Warmup bounds computation
        
        vis_mask = np.ones(n, dtype=np.bool_)
        cam = np.zeros(3, dtype=np.float64)
        fwd = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        compute_visibility_points(pos, cam, fwd, right, up, 1.0, 1.0, 100.0, vis_mask, n)
    
    def _init_vbos(self):
        """Initialize VBOs for rendering."""
        if self._vbos_initialized:
            return
        
        try:
            # Use float32 for GPU
            pos_f32 = self.positions.astype(np.float32)
            self._vbo_positions = vbo.VBO(pos_f32, usage=GL_DYNAMIC_DRAW)
            self._vbo_colors = vbo.VBO(self.colors, usage=GL_DYNAMIC_DRAW)
            self._vbos_initialized = True
        except Exception as e:
            print(f"[NBody] VBO init failed: {e}")
            self._vbos_initialized = False
    
    def update(self, dt: float):
        """Update simulation one timestep."""
        # Cap dt for stability
        dt = min(dt, 0.02)
        
        if self._use_gpu:
            self._update_gpu(dt)
        else:
            self._update_cpu(dt)
    
    def _update_gpu(self, dt: float):
        """GPU-accelerated update."""
        # Run physics on GPU
        self._gpu_sim.step(dt)
        self._gpu_sim.compute_colors(self.max_speed_color)
        
        # Copy back to CPU for rendering (positions and colors)
        self.positions = self._gpu_sim.get_positions().astype(np.float64)
        self.colors = self._gpu_sim.get_colors()
    
    def _update_cpu(self, dt: float):
        """CPU update using Barnes-Hut algorithm."""
        # Compute dynamic bounds from particle positions (no fixed boundaries)
        self.current_bounds = compute_bounds(self.positions, self.num_bodies)
        
        # Rebuild octree each frame with dynamic bounds
        self._node_children.fill(-1)
        self._node_body_idx.fill(-1)
        self._node_is_leaf.fill(True)
        
        self._num_tree_nodes = build_octree(
            self.positions,
            self.masses,
            self.num_bodies,
            self.current_bounds,
            self._node_centers,
            self._node_half_sizes,
            self._node_masses,
            self._node_com,
            self._node_children,
            self._node_body_idx,
            self._node_is_leaf
        )
        
        # Compute forces using Barnes-Hut
        compute_forces_barnes_hut(
            self.positions,
            self.masses,
            self.accelerations,
            self._node_centers,
            self._node_half_sizes,
            self._node_masses,
            self._node_com,
            self._node_children,
            self._node_body_idx,
            self._node_is_leaf,
            self._num_tree_nodes,
            self.num_bodies,
            self.theta,
            self.G,
            self.softening
        )
        
        # Update positions and velocities - no boundaries!
        update_positions_velocities(
            self.positions,
            self.velocities,
            self.accelerations,
            self.damping,
            dt,
            self.num_bodies
        )
        
        # Update colors based on velocity
        compute_colors_by_velocity(
            self.velocities,
            self.colors,
            self.num_bodies,
            self.max_speed_color
        )
    
    def _compute_visibility(self, cam_pos, cam_forward, cam_right, cam_up, fov_v, aspect):
        """Compute which bodies are visible."""
        self._cam_pos[:] = cam_pos
        self._cam_forward[:] = cam_forward
        self._cam_right[:] = cam_right
        self._cam_up[:] = cam_up
        
        half_fov_v = fov_v / 2
        half_fov_h = math.atan(math.tan(half_fov_v) * aspect)
        
        compute_visibility_points(
            self.positions,
            self._cam_pos,
            self._cam_forward,
            self._cam_right,
            self._cam_up,
            math.tan(half_fov_h),
            math.tan(half_fov_v),
            self.fog_end,
            self._visible_mask,
            self.num_bodies
        )
        
        self._visible_count = np.sum(self._visible_mask)
    
    def draw(self, cam_pos=None, cam_forward=None, cam_right=None, cam_up=None,
             fov=None, aspect=None):
        """Render bodies as point sprites."""
        if not self._vbos_initialized:
            self._init_vbos()
        
        # Compute visibility
        if cam_pos is not None:
            fov_rad = math.radians(fov) if fov else math.radians(75)
            aspect = aspect if aspect else (16/9)
            self._compute_visibility(cam_pos, cam_forward, cam_right, cam_up, fov_rad, aspect)
        else:
            self._visible_mask[:] = True
            self._visible_count = self.num_bodies
        
        # Setup point rendering
        glEnable(GL_POINT_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)  # Additive blending for glow effect
        glPointSize(self.point_size)
        
        # Get visible positions and colors
        visible_pos = self.positions[self._visible_mask].astype(np.float32)
        visible_colors = self.colors[self._visible_mask]
        
        if len(visible_pos) == 0:
            return
        
        if self._vbos_initialized and self._vbo_positions is not None:
            # VBO path
            self._vbo_positions.set_array(visible_pos)
            self._vbo_colors.set_array(visible_colors)
            
            self._vbo_positions.bind()
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, None)
            
            self._vbo_colors.bind()
            glEnableClientState(GL_COLOR_ARRAY)
            glColorPointer(3, GL_FLOAT, 0, None)
            
            glDrawArrays(GL_POINTS, 0, len(visible_pos))
            
            self._vbo_positions.unbind()
            self._vbo_colors.unbind()
            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_COLOR_ARRAY)
        else:
            # Fallback
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_COLOR_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, visible_pos)
            glColorPointer(3, GL_FLOAT, 0, visible_colors)
            glDrawArrays(GL_POINTS, 0, len(visible_pos))
            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_COLOR_ARRAY)
        
        glDisable(GL_BLEND)
        glDisable(GL_POINT_SMOOTH)

