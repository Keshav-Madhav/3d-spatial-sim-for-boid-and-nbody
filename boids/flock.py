"""Flock management and rendering - maximum performance with spatial hashing, Numba JIT, and VBOs."""

import math
import numpy as np
from numba import njit, prange
from OpenGL.GL import *
from OpenGL.arrays import vbo

from config import boids as config


# ============================================================================
# NUMBA JIT-COMPILED SPATIAL GRID FUNCTIONS
# ============================================================================

@njit(cache=True)
def get_cell_index(x: float, y: float, z: float, cell_size: float, grid_dim: int, offset: float) -> int:
    """Convert 3D position to 1D cell index."""
    cx = int((x + offset) / cell_size)
    cy = int((y + offset) / cell_size)
    cz = int((z + offset) / cell_size)
    
    cx = max(0, min(cx, grid_dim - 1))
    cy = max(0, min(cy, grid_dim - 1))
    cz = max(0, min(cz, grid_dim - 1))
    
    return cx + cy * grid_dim + cz * grid_dim * grid_dim


@njit(parallel=True, cache=True)
def assign_cells(
    positions: np.ndarray,
    cell_indices: np.ndarray,
    cell_size: float,
    grid_dim: int,
    offset: float,
    num_boids: int
):
    """Assign each boid to a cell."""
    for i in prange(num_boids):
        cell_indices[i] = get_cell_index(
            positions[i, 0], positions[i, 1], positions[i, 2],
            cell_size, grid_dim, offset
        )


@njit(cache=True)
def build_cell_lists(
    cell_indices: np.ndarray,
    sorted_indices: np.ndarray,
    cell_starts: np.ndarray,
    cell_counts: np.ndarray,
    num_boids: int,
    num_cells: int
):
    """Build cell start indices and counts after sorting."""
    for i in range(num_cells):
        cell_starts[i] = -1
        cell_counts[i] = 0
    
    for i in range(num_boids):
        cell = cell_indices[sorted_indices[i]]
        if cell_starts[cell] == -1:
            cell_starts[cell] = i
        cell_counts[cell] += 1


@njit(parallel=True, fastmath=True, cache=True)
def compute_flocking_spatial(
    positions: np.ndarray,
    velocities: np.ndarray,
    colors: np.ndarray,
    sorted_indices: np.ndarray,
    cell_starts: np.ndarray,
    cell_counts: np.ndarray,
    separation_forces: np.ndarray,
    alignment_forces: np.ndarray,
    cohesion_forces: np.ndarray,
    avg_colors: np.ndarray,
    cell_size: float,
    grid_dim: int,
    offset: float,
    perception_radius: float,
    separation_radius: float,
    separation_weight: float,
    alignment_weight: float,
    cohesion_weight: float,
    max_speed: float,
    max_force: float,
    num_boids: int
):
    """Numba JIT-compiled flocking with spatial grid acceleration."""
    perception_sq = perception_radius * perception_radius
    separation_sq = separation_radius * separation_radius
    cell_range = int(np.ceil(perception_radius / cell_size))
    
    for i in prange(num_boids):
        pos_i = positions[i]
        vel_i = velocities[i]
        
        cx = int((pos_i[0] + offset) / cell_size)
        cy = int((pos_i[1] + offset) / cell_size)
        cz = int((pos_i[2] + offset) / cell_size)
        
        cx = max(0, min(cx, grid_dim - 1))
        cy = max(0, min(cy, grid_dim - 1))
        cz = max(0, min(cz, grid_dim - 1))
        
        sep_x, sep_y, sep_z = 0.0, 0.0, 0.0
        align_x, align_y, align_z = 0.0, 0.0, 0.0
        coh_x, coh_y, coh_z = 0.0, 0.0, 0.0
        col_r, col_g, col_b = 0.0, 0.0, 0.0
        
        sep_count = 0
        neighbor_count = 0
        
        for dcx in range(-cell_range, cell_range + 1):
            ncx = cx + dcx
            if ncx < 0 or ncx >= grid_dim:
                continue
                
            for dcy in range(-cell_range, cell_range + 1):
                ncy = cy + dcy
                if ncy < 0 or ncy >= grid_dim:
                    continue
                    
                for dcz in range(-cell_range, cell_range + 1):
                    ncz = cz + dcz
                    if ncz < 0 or ncz >= grid_dim:
                        continue
                    
                    cell_idx = ncx + ncy * grid_dim + ncz * grid_dim * grid_dim
                    
                    start = cell_starts[cell_idx]
                    if start == -1:
                        continue
                    
                    count = cell_counts[cell_idx]
                    
                    for k in range(count):
                        j = sorted_indices[start + k]
                        if i == j:
                            continue
                        
                        dx = pos_i[0] - positions[j, 0]
                        dy = pos_i[1] - positions[j, 1]
                        dz = pos_i[2] - positions[j, 2]
                        dist_sq = dx * dx + dy * dy + dz * dz
                        
                        if dist_sq < perception_sq and dist_sq > 0.0001:
                            dist = math.sqrt(dist_sq)
                            
                            if dist_sq < separation_sq:
                                inv_dist = 1.0 / dist
                                sep_x += dx * inv_dist / dist
                                sep_y += dy * inv_dist / dist
                                sep_z += dz * inv_dist / dist
                                sep_count += 1
                            
                            align_x += velocities[j, 0]
                            align_y += velocities[j, 1]
                            align_z += velocities[j, 2]
                            
                            coh_x += positions[j, 0]
                            coh_y += positions[j, 1]
                            coh_z += positions[j, 2]
                            
                            col_r += colors[j, 0]
                            col_g += colors[j, 1]
                            col_b += colors[j, 2]
                            
                            neighbor_count += 1
        
        if sep_count > 0:
            sep_x /= sep_count
            sep_y /= sep_count
            sep_z /= sep_count
            
            sep_mag = math.sqrt(sep_x * sep_x + sep_y * sep_y + sep_z * sep_z)
            if sep_mag > 0:
                sep_x = (sep_x / sep_mag) * max_speed - vel_i[0]
                sep_y = (sep_y / sep_mag) * max_speed - vel_i[1]
                sep_z = (sep_z / sep_mag) * max_speed - vel_i[2]
                
                sep_mag = math.sqrt(sep_x * sep_x + sep_y * sep_y + sep_z * sep_z)
                if sep_mag > max_force:
                    sep_x = (sep_x / sep_mag) * max_force
                    sep_y = (sep_y / sep_mag) * max_force
                    sep_z = (sep_z / sep_mag) * max_force
                
                separation_forces[i, 0] = sep_x * separation_weight
                separation_forces[i, 1] = sep_y * separation_weight
                separation_forces[i, 2] = sep_z * separation_weight
        
        if neighbor_count > 0:
            align_x /= neighbor_count
            align_y /= neighbor_count
            align_z /= neighbor_count
            
            align_mag = math.sqrt(align_x * align_x + align_y * align_y + align_z * align_z)
            if align_mag > 0:
                align_x = (align_x / align_mag) * max_speed - vel_i[0]
                align_y = (align_y / align_mag) * max_speed - vel_i[1]
                align_z = (align_z / align_mag) * max_speed - vel_i[2]
                
                align_mag = math.sqrt(align_x * align_x + align_y * align_y + align_z * align_z)
                if align_mag > max_force:
                    align_x = (align_x / align_mag) * max_force
                    align_y = (align_y / align_mag) * max_force
                    align_z = (align_z / align_mag) * max_force
                
                alignment_forces[i, 0] = align_x * alignment_weight
                alignment_forces[i, 1] = align_y * alignment_weight
                alignment_forces[i, 2] = align_z * alignment_weight
            
            coh_x = coh_x / neighbor_count - pos_i[0]
            coh_y = coh_y / neighbor_count - pos_i[1]
            coh_z = coh_z / neighbor_count - pos_i[2]
            
            coh_mag = math.sqrt(coh_x * coh_x + coh_y * coh_y + coh_z * coh_z)
            if coh_mag > 0:
                coh_x = (coh_x / coh_mag) * max_speed - vel_i[0]
                coh_y = (coh_y / coh_mag) * max_speed - vel_i[1]
                coh_z = (coh_z / coh_mag) * max_speed - vel_i[2]
                
                coh_mag = math.sqrt(coh_x * coh_x + coh_y * coh_y + coh_z * coh_z)
                if coh_mag > max_force:
                    coh_x = (coh_x / coh_mag) * max_force
                    coh_y = (coh_y / coh_mag) * max_force
                    coh_z = (coh_z / coh_mag) * max_force
                
                cohesion_forces[i, 0] = coh_x * cohesion_weight
                cohesion_forces[i, 1] = coh_y * cohesion_weight
                cohesion_forces[i, 2] = coh_z * cohesion_weight
            
            avg_colors[i, 0] = (col_r + colors[i, 0]) / (neighbor_count + 1)
            avg_colors[i, 1] = (col_g + colors[i, 1]) / (neighbor_count + 1)
            avg_colors[i, 2] = (col_b + colors[i, 2]) / (neighbor_count + 1)


@njit(parallel=True, fastmath=True, cache=True)
def update_physics_numba(
    positions: np.ndarray,
    velocities: np.ndarray,
    colors: np.ndarray,
    sep_forces: np.ndarray,
    align_forces: np.ndarray,
    coh_forces: np.ndarray,
    avg_colors: np.ndarray,
    bounds: float,
    margin: float,
    wall_force: float,
    max_speed: float,
    color_blend: float,
    dt: float,
    num_boids: int
):
    """Numba JIT-compiled physics update."""
    for i in prange(num_boids):
        ax = sep_forces[i, 0] + align_forces[i, 0] + coh_forces[i, 0]
        ay = sep_forces[i, 1] + align_forces[i, 1] + coh_forces[i, 1]
        az = sep_forces[i, 2] + align_forces[i, 2] + coh_forces[i, 2]
        
        for dim in range(3):
            pos = positions[i, dim]
            
            dist_pos = pos - (bounds - margin)
            if dist_pos > 0:
                strength = min(dist_pos / margin * 2.0, 1.0)
                if dim == 0:
                    ax -= strength * wall_force
                elif dim == 1:
                    ay -= strength * wall_force
                else:
                    az -= strength * wall_force
            
            dist_neg = (-bounds + margin) - pos
            if dist_neg > 0:
                strength = min(dist_neg / margin * 2.0, 1.0)
                if dim == 0:
                    ax += strength * wall_force
                elif dim == 1:
                    ay += strength * wall_force
                else:
                    az += strength * wall_force
        
        velocities[i, 0] += ax * dt
        velocities[i, 1] += ay * dt
        velocities[i, 2] += az * dt
        
        speed = math.sqrt(
            velocities[i, 0] ** 2 + 
            velocities[i, 1] ** 2 + 
            velocities[i, 2] ** 2
        )
        if speed > max_speed:
            scale = max_speed / speed
            velocities[i, 0] *= scale
            velocities[i, 1] *= scale
            velocities[i, 2] *= scale
        
        positions[i, 0] += velocities[i, 0] * dt
        positions[i, 1] += velocities[i, 1] * dt
        positions[i, 2] += velocities[i, 2] * dt
        
        colors[i, 0] += (avg_colors[i, 0] - colors[i, 0]) * color_blend
        colors[i, 1] += (avg_colors[i, 1] - colors[i, 1]) * color_blend
        colors[i, 2] += (avg_colors[i, 2] - colors[i, 2]) * color_blend


@njit(parallel=True, fastmath=True, cache=True)
def compute_visibility_numba(
    positions: np.ndarray,
    cam_pos: np.ndarray,
    cam_forward: np.ndarray,
    cam_right: np.ndarray,
    cam_up: np.ndarray,
    tan_h: float,
    tan_v: float,
    fog_end: float,
    visible_mask: np.ndarray,
    num_boids: int
):
    """Numba JIT-compiled frustum culling."""
    for i in prange(num_boids):
        dx = positions[i, 0] - cam_pos[0]
        dy = positions[i, 1] - cam_pos[1]
        dz = positions[i, 2] - cam_pos[2]
        
        # Project onto camera axes
        z = dx * cam_forward[0] + dy * cam_forward[1] + dz * cam_forward[2]
        
        # Quick depth check first (most common rejection)
        if z < 0.5 or z > fog_end:
            visible_mask[i] = False
            continue
        
        x = dx * cam_right[0] + dy * cam_right[1] + dz * cam_right[2]
        y = dx * cam_up[0] + dy * cam_up[1] + dz * cam_up[2]
        
        # Frustum check
        half_width = z * tan_h
        half_height = z * tan_v
        
        if abs(x) < half_width and abs(y) < half_height:
            visible_mask[i] = True
        else:
            visible_mask[i] = False


@njit(parallel=True, fastmath=True, cache=True)
def build_vertices_numba(
    positions: np.ndarray,
    velocities: np.ndarray,
    colors: np.ndarray,
    visible_indices: np.ndarray,
    vertices: np.ndarray,
    vert_colors: np.ndarray,
    cone_length: float,
    cone_radius: float,
    num_visible: int
):
    """Numba JIT-compiled vertex building."""
    world_up_x, world_up_y, world_up_z = 0.0, 1.0, 0.0
    world_right_x, world_right_y, world_right_z = 1.0, 0.0, 0.0
    
    for idx in prange(num_visible):
        i = visible_indices[idx]
        
        # Position
        px, py, pz = positions[i, 0], positions[i, 1], positions[i, 2]
        
        # Velocity -> forward direction
        vx, vy, vz = velocities[i, 0], velocities[i, 1], velocities[i, 2]
        speed = math.sqrt(vx * vx + vy * vy + vz * vz)
        if speed < 0.0001:
            speed = 0.0001
        fx, fy, fz = vx / speed, vy / speed, vz / speed
        
        # Right = forward x world_up
        rx = fy * world_up_z - fz * world_up_y
        ry = fz * world_up_x - fx * world_up_z
        rz = fx * world_up_y - fy * world_up_x
        
        r_len = math.sqrt(rx * rx + ry * ry + rz * rz)
        if r_len < 0.1:
            # Use world_right instead
            rx = fy * world_right_z - fz * world_right_y
            ry = fz * world_right_x - fx * world_right_z
            rz = fx * world_right_y - fy * world_right_x
            r_len = math.sqrt(rx * rx + ry * ry + rz * rz)
        
        if r_len > 0.0001:
            rx /= r_len
            ry /= r_len
            rz /= r_len
        
        # Up = right x forward
        ux = ry * fz - rz * fy
        uy = rz * fx - rx * fz
        uz = rx * fy - ry * fx
        
        # Cone tip
        tip_x = px + fx * cone_length
        tip_y = py + fy * cone_length
        tip_z = pz + fz * cone_length
        
        # Base points
        r = cone_radius
        base_r_x, base_r_y, base_r_z = px + rx * r, py + ry * r, pz + rz * r
        base_l_x, base_l_y, base_l_z = px - rx * r, py - ry * r, pz - rz * r
        base_u_x, base_u_y, base_u_z = px + ux * r, py + uy * r, pz + uz * r
        base_d_x, base_d_y, base_d_z = px - ux * r, py - uy * r, pz - uz * r
        
        # Color
        cr, cg, cb = colors[i, 0], colors[i, 1], colors[i, 2]
        
        # Write vertices (6 per boid: 2 triangles)
        base = idx * 6
        
        # Triangle 1: tip, base_r, base_l
        vertices[base, 0] = tip_x
        vertices[base, 1] = tip_y
        vertices[base, 2] = tip_z
        vertices[base + 1, 0] = base_r_x
        vertices[base + 1, 1] = base_r_y
        vertices[base + 1, 2] = base_r_z
        vertices[base + 2, 0] = base_l_x
        vertices[base + 2, 1] = base_l_y
        vertices[base + 2, 2] = base_l_z
        
        # Triangle 2: tip, base_u, base_d
        vertices[base + 3, 0] = tip_x
        vertices[base + 3, 1] = tip_y
        vertices[base + 3, 2] = tip_z
        vertices[base + 4, 0] = base_u_x
        vertices[base + 4, 1] = base_u_y
        vertices[base + 4, 2] = base_u_z
        vertices[base + 5, 0] = base_d_x
        vertices[base + 5, 1] = base_d_y
        vertices[base + 5, 2] = base_d_z
        
        # Colors for all 6 vertices
        for v in range(6):
            vert_colors[base + v, 0] = cr
            vert_colors[base + v, 1] = cg
            vert_colors[base + v, 2] = cb


# ============================================================================
# FLOCK CLASS
# ============================================================================

class Flock:
    """
    Maximum performance flock with spatial hashing, Numba JIT, and VBO rendering.
    """
    
    def __init__(self, num_boids: int = 1000):
        self.num_boids = num_boids
        self.bounds = np.float64(config.BOIDS["bounds"])
        self.wall_margin = np.float64(config.BOIDS["wall_margin"])
        self.wall_weight = np.float64(config.BOIDS["wall_weight"])
        self.max_speed = np.float64(config.BOIDS["max_speed"])
        self.max_force = np.float64(config.BOIDS["max_force"])
        self.cone_length = np.float32(config.BOIDS["size"])
        self.cone_radius = np.float32(config.BOIDS["size"] * 0.35)
        
        # Flocking parameters
        self.perception_radius = np.float64(config.BOIDS["perception_radius"])
        self.separation_radius = np.float64(config.BOIDS["separation_radius"])
        self.separation_weight = np.float64(config.BOIDS["separation_weight"])
        self.alignment_weight = np.float64(config.BOIDS["alignment_weight"])
        self.cohesion_weight = np.float64(config.BOIDS["cohesion_weight"])
        self.color_blend_rate = np.float64(config.BOIDS["color_blend_rate"])
        
        # Spatial grid parameters
        self.cell_size = float(self.perception_radius)
        self.grid_dim = int(np.ceil(self.bounds * 2 / self.cell_size)) + 2
        self.num_cells = self.grid_dim ** 3
        self.grid_offset = float(self.bounds + self.cell_size)
        
        # Culling - use full far_clip now
        self.fog_end = float(config.CAMERA["far_clip"])
        self.fov_margin = 1.15
        
        # Boid data (float64 for physics accuracy)
        self.positions = ((np.random.rand(num_boids, 3) - 0.5) * 2 * self.bounds).astype(np.float64)
        self.velocities = ((np.random.rand(num_boids, 3) - 0.5) * self.max_speed).astype(np.float64)
        self.colors = self._generate_colors(num_boids).astype(np.float64)
        
        # Spatial grid arrays
        self._cell_indices = np.zeros(num_boids, dtype=np.int32)
        self._sorted_indices = np.arange(num_boids, dtype=np.int32)
        self._cell_starts = np.zeros(self.num_cells, dtype=np.int32)
        self._cell_counts = np.zeros(self.num_cells, dtype=np.int32)
        
        # Force arrays
        self._sep_forces = np.zeros((num_boids, 3), dtype=np.float64)
        self._align_forces = np.zeros((num_boids, 3), dtype=np.float64)
        self._coh_forces = np.zeros((num_boids, 3), dtype=np.float64)
        self._avg_colors = np.zeros((num_boids, 3), dtype=np.float64)
        
        # Visibility
        self._visible_mask = np.ones(num_boids, dtype=np.bool_)
        self._visible_indices = np.arange(num_boids, dtype=np.int32)
        self._visible_count = num_boids
        
        # Vertex data (float32 for GPU)
        self.verts_per_boid = 6
        self._vertices = np.zeros((num_boids * self.verts_per_boid, 3), dtype=np.float32)
        self._vert_colors = np.zeros((num_boids * self.verts_per_boid, 3), dtype=np.float32)
        
        # VBOs for GPU-side storage
        self._vbo_vertices = None
        self._vbo_colors = None
        self._vbos_initialized = False
        
        # Camera arrays for Numba
        self._cam_pos = np.zeros(3, dtype=np.float64)
        self._cam_forward = np.zeros(3, dtype=np.float64)
        self._cam_right = np.zeros(3, dtype=np.float64)
        self._cam_up = np.zeros(3, dtype=np.float64)
        
        # Warm up Numba
        self._warmup_numba()
    
    def _init_vbos(self):
        """Initialize VBOs for fast GPU rendering."""
        if self._vbos_initialized:
            return
        
        try:
            self._vbo_vertices = vbo.VBO(self._vertices, usage=GL_DYNAMIC_DRAW)
            self._vbo_colors = vbo.VBO(self._vert_colors, usage=GL_DYNAMIC_DRAW)
            self._vbos_initialized = True
        except Exception:
            # Fallback to non-VBO rendering
            self._vbos_initialized = False
    
    def _warmup_numba(self):
        """Pre-compile Numba functions."""
        n = 100
        pos = np.random.rand(n, 3).astype(np.float64) * 10
        vel = np.random.rand(n, 3).astype(np.float64)
        col = np.random.rand(n, 3).astype(np.float64)
        f1 = np.zeros((n, 3), dtype=np.float64)
        f2 = np.zeros((n, 3), dtype=np.float64)
        f3 = np.zeros((n, 3), dtype=np.float64)
        avg = col.copy()
        
        cell_idx = np.zeros(n, dtype=np.int32)
        sorted_idx = np.arange(n, dtype=np.int32)
        cell_starts = np.zeros(1000, dtype=np.int32)
        cell_counts = np.zeros(1000, dtype=np.int32)
        
        assign_cells(pos, cell_idx, 5.0, 10, 50.0, n)
        order = np.argsort(cell_idx)
        sorted_idx[:] = order
        build_cell_lists(cell_idx, sorted_idx, cell_starts, cell_counts, n, 1000)
        
        compute_flocking_spatial(
            pos, vel, col, sorted_idx, cell_starts, cell_counts,
            f1, f2, f3, avg,
            5.0, 10, 50.0, 5.0, 2.0, 1.0, 1.0, 1.0, 10.0, 10.0, n
        )
        
        update_physics_numba(
            pos, vel, col, f1, f2, f3, avg,
            50.0, 5.0, 10.0, 10.0, 0.1, 0.016, n
        )
        
        # Visibility warmup
        vis_mask = np.ones(n, dtype=np.bool_)
        cam = np.zeros(3, dtype=np.float64)
        fwd = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        compute_visibility_numba(pos, cam, fwd, right, up, 1.0, 1.0, 100.0, vis_mask, n)
        
        # Vertex building warmup
        vis_idx = np.arange(n, dtype=np.int32)
        verts = np.zeros((n * 6, 3), dtype=np.float32)
        vcols = np.zeros((n * 6, 3), dtype=np.float32)
        build_vertices_numba(pos, vel, col, vis_idx, verts, vcols, 1.0, 0.3, n)
    
    def _generate_colors(self, count: int) -> np.ndarray:
        """Generate rainbow colors."""
        hues = np.linspace(0, 1, count, endpoint=False, dtype=np.float32)
        np.random.shuffle(hues)
        
        s, v = 0.9, 1.0
        h6 = hues * 6.0
        i = (h6).astype(np.int32) % 6
        f = h6 - np.floor(h6)
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        
        colors = np.zeros((count, 3), dtype=np.float32)
        
        for idx, (rv, gv, bv) in enumerate([(v, 't', p), ('q', v, p), (p, v, 't'), (p, 'q', v), ('t', p, v), (v, p, 'q')]):
            mask = (i == idx)
            colors[mask, 0] = v if rv == v else (p if rv == p else (t[mask] if rv == 't' else q[mask]))
            colors[mask, 1] = v if gv == v else (p if gv == p else (t[mask] if gv == 't' else q[mask]))
            colors[mask, 2] = v if bv == v else (p if bv == p else (t[mask] if bv == 't' else q[mask]))
        
        return colors
    
    def _build_spatial_grid(self):
        """Build spatial grid for efficient neighbor queries."""
        assign_cells(
            self.positions, self._cell_indices,
            self.cell_size, self.grid_dim, self.grid_offset,
            self.num_boids
        )
        
        order = np.argsort(self._cell_indices)
        self._sorted_indices[:] = order
        
        build_cell_lists(
            self._cell_indices, self._sorted_indices,
            self._cell_starts, self._cell_counts,
            self.num_boids, self.num_cells
        )
    
    def update(self, dt: float):
        """Update all boids using spatial grid + Numba acceleration."""
        dt64 = np.float64(dt)
        
        self._build_spatial_grid()
        
        self._sep_forces.fill(0)
        self._align_forces.fill(0)
        self._coh_forces.fill(0)
        np.copyto(self._avg_colors, self.colors)
        
        compute_flocking_spatial(
            self.positions,
            self.velocities,
            self.colors,
            self._sorted_indices,
            self._cell_starts,
            self._cell_counts,
            self._sep_forces,
            self._align_forces,
            self._coh_forces,
            self._avg_colors,
            self.cell_size,
            self.grid_dim,
            self.grid_offset,
            float(self.perception_radius),
            float(self.separation_radius),
            float(self.separation_weight),
            float(self.alignment_weight),
            float(self.cohesion_weight),
            float(self.max_speed),
            float(self.max_force),
            self.num_boids
        )
        
        blend = min(1.0, float(self.color_blend_rate) * dt64)
        update_physics_numba(
            self.positions,
            self.velocities,
            self.colors,
            self._sep_forces,
            self._align_forces,
            self._coh_forces,
            self._avg_colors,
            float(self.bounds),
            float(self.wall_margin),
            float(self.max_force * self.wall_weight),
            float(self.max_speed),
            blend,
            dt64,
            self.num_boids
        )
    
    def _compute_visibility(self, cam_pos, cam_forward, cam_right, cam_up, fov_v, aspect):
        """Compute visibility using Numba-accelerated frustum culling."""
        # Copy to contiguous arrays
        self._cam_pos[:] = cam_pos
        self._cam_forward[:] = cam_forward
        self._cam_right[:] = cam_right
        self._cam_up[:] = cam_up
        
        half_fov_v = (fov_v / 2) * self.fov_margin
        half_fov_h = math.atan(math.tan(half_fov_v) * aspect)
        
        tan_h = math.tan(half_fov_h)
        tan_v = math.tan(half_fov_v)
        
        compute_visibility_numba(
            self.positions,
            self._cam_pos,
            self._cam_forward,
            self._cam_right,
            self._cam_up,
            tan_h,
            tan_v,
            self.fog_end,
            self._visible_mask,
            self.num_boids
        )
        
        # Get visible indices
        self._visible_indices = np.where(self._visible_mask)[0].astype(np.int32)
        self._visible_count = len(self._visible_indices)
    
    def _build_vertices(self):
        """Build vertex data using Numba."""
        if self._visible_count == 0:
            return 0
        
        build_vertices_numba(
            self.positions,
            self.velocities,
            self.colors,
            self._visible_indices,
            self._vertices,
            self._vert_colors,
            float(self.cone_length),
            float(self.cone_radius),
            self._visible_count
        )
        
        return self._visible_count * self.verts_per_boid
    
    def draw(self, cam_pos=None, cam_forward=None, cam_right=None, cam_up=None, 
             fov=None, aspect=None):
        """Render visible boids using VBOs."""
        # Initialize VBOs on first draw
        if not self._vbos_initialized:
            self._init_vbos()
        
        # Compute visibility
        if cam_pos is None:
            self._visible_mask[:] = True
            self._visible_indices = np.arange(self.num_boids, dtype=np.int32)
            self._visible_count = self.num_boids
        else:
            fov_rad = math.radians(fov) if fov else math.radians(75)
            aspect = aspect if aspect else (16/9)
            self._compute_visibility(cam_pos, cam_forward, cam_right, cam_up, fov_rad, aspect)
        
        if self._visible_count == 0:
            return
        
        # Build vertices
        total_verts = self._build_vertices()
        
        if self._vbos_initialized and self._vbo_vertices is not None:
            # VBO rendering path (faster)
            self._vbo_vertices.set_array(self._vertices[:total_verts])
            self._vbo_colors.set_array(self._vert_colors[:total_verts])
            
            self._vbo_vertices.bind()
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, None)
            
            self._vbo_colors.bind()
            glEnableClientState(GL_COLOR_ARRAY)
            glColorPointer(3, GL_FLOAT, 0, None)
            
            glDrawArrays(GL_TRIANGLES, 0, total_verts)
            
            self._vbo_vertices.unbind()
            self._vbo_colors.unbind()
            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_COLOR_ARRAY)
        else:
            # Fallback to immediate mode
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_COLOR_ARRAY)
            
            glVertexPointer(3, GL_FLOAT, 0, self._vertices[:total_verts])
            glColorPointer(3, GL_FLOAT, 0, self._vert_colors[:total_verts])
            glDrawArrays(GL_TRIANGLES, 0, total_verts)
            
            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_COLOR_ARRAY)
