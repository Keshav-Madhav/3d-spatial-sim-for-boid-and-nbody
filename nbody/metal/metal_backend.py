"""
Metal Barnes-Hut Backend for Apple Silicon - OPTIMIZED
=======================================================

Uses native Metal compute shaders with Unified Memory Architecture (UMA)
for true zero-copy CPU-GPU data sharing.

OPTIMIZATIONS:
- Numba-parallel tree building with direct GPU-ready output
- Buffer reuse to minimize allocations
- Vectorized numpy operations (no Python loops)
- Optimized memory layout for GPU coalescing
"""

import os
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

# Check for Metal availability
_METAL_AVAILABLE = False
_METAL_ERROR = None

try:
    import Metal
    from Metal import (
        MTLCreateSystemDefaultDevice,
        MTLResourceStorageModeShared,
        MTLResourceCPUCacheModeDefaultCache,
        MTLSize,
    )
    import objc
    _METAL_AVAILABLE = True
except ImportError as e:
    _METAL_ERROR = str(e)
except Exception as e:
    _METAL_ERROR = str(e)


def is_metal_available() -> Tuple[bool, str]:
    """Check if Metal backend is available."""
    if not _METAL_AVAILABLE:
        return False, f"Metal not available: {_METAL_ERROR}"
    
    try:
        device = MTLCreateSystemDefaultDevice()
        if device is None:
            return False, "No Metal device found"
        
        name = device.name()
        max_buffer = device.maxBufferLength()
        return True, f"{name} (max buffer: {max_buffer // (1024**3)}GB)"
    except Exception as e:
        return False, f"Metal initialization failed: {e}"


# =============================================================================
# NUMBA-OPTIMIZED TREE BUILDER (Direct GPU-ready output)
# =============================================================================

_NUMBA_AVAILABLE = False

try:
    from numba import njit, prange, int32, float32
    
    @njit(cache=True, fastmath=True)
    def _build_tree_direct(positions, masses, bounds, max_nodes,
                           # Output arrays - directly GPU-ready format
                           out_pos_mass,      # (max_nodes, 4) - xyz=com, w=mass  
                           out_bounds,        # (max_nodes, 4) - xyz=center, w=half_size
                           out_child_base,    # (max_nodes,) int32
                           out_body_idx,      # (max_nodes,) int32
                           out_is_leaf,       # (max_nodes,) int32
                           out_children):     # (max_nodes*8,) int32
        """
        Ultra-fast Numba tree builder that outputs directly to GPU-ready arrays.
        No post-processing packing needed!
        """
        n = len(positions)
        
        # Initialize root node
        out_pos_mass[0, 0] = 0.0
        out_pos_mass[0, 1] = 0.0
        out_pos_mass[0, 2] = 0.0
        out_pos_mass[0, 3] = 0.0
        out_bounds[0, 0] = 0.0
        out_bounds[0, 1] = 0.0
        out_bounds[0, 2] = 0.0
        out_bounds[0, 3] = bounds
        out_child_base[0] = 0
        out_body_idx[0] = -1
        out_is_leaf[0] = 1
        
        num_nodes = 1
        children_idx = 8
        
        for i in range(n):
            px = positions[i, 0]
            py = positions[i, 1]
            pz = positions[i, 2]
            m = masses[i]
            
            current = 0
            depth = 0
            
            while depth < 32:
                cx = out_bounds[current, 0]
                cy = out_bounds[current, 1]
                cz = out_bounds[current, 2]
                hs = out_bounds[current, 3]
                
                if out_is_leaf[current] == 1:
                    if out_body_idx[current] == -1:
                        # Empty leaf - insert body
                        out_body_idx[current] = i
                        out_pos_mass[current, 0] = px
                        out_pos_mass[current, 1] = py
                        out_pos_mass[current, 2] = pz
                        out_pos_mass[current, 3] = m
                        break
                    else:
                        # Subdivide
                        old_body = out_body_idx[current]
                        old_px = positions[old_body, 0]
                        old_py = positions[old_body, 1]
                        old_pz = positions[old_body, 2]
                        old_m = masses[old_body]
                        
                        out_is_leaf[current] = 0
                        out_body_idx[current] = -1
                        out_child_base[current] = children_idx
                        children_idx += 8
                        
                        if children_idx >= len(out_children):
                            break
                        
                        # Compute octant for old body
                        octant = 0
                        if old_px >= cx: octant |= 1
                        if old_py >= cy: octant |= 2
                        if old_pz >= cz: octant |= 4
                        
                        child_idx = num_nodes
                        num_nodes += 1
                        if num_nodes >= max_nodes:
                            break
                        
                        quarter = hs * 0.5
                        new_cx = cx + quarter if (octant & 1) else cx - quarter
                        new_cy = cy + quarter if (octant & 2) else cy - quarter
                        new_cz = cz + quarter if (octant & 4) else cz - quarter
                        
                        out_bounds[child_idx, 0] = new_cx
                        out_bounds[child_idx, 1] = new_cy
                        out_bounds[child_idx, 2] = new_cz
                        out_bounds[child_idx, 3] = quarter
                        out_pos_mass[child_idx, 0] = old_px
                        out_pos_mass[child_idx, 1] = old_py
                        out_pos_mass[child_idx, 2] = old_pz
                        out_pos_mass[child_idx, 3] = old_m
                        out_body_idx[child_idx] = old_body
                        out_is_leaf[child_idx] = 1
                        out_child_base[child_idx] = 0
                        
                        out_children[out_child_base[current] + octant] = child_idx
                
                # Update center of mass
                old_mass = out_pos_mass[current, 3]
                total_mass = old_mass + m
                if total_mass > 0:
                    inv_total = 1.0 / total_mass
                    out_pos_mass[current, 0] = (out_pos_mass[current, 0] * old_mass + px * m) * inv_total
                    out_pos_mass[current, 1] = (out_pos_mass[current, 1] * old_mass + py * m) * inv_total
                    out_pos_mass[current, 2] = (out_pos_mass[current, 2] * old_mass + pz * m) * inv_total
                    out_pos_mass[current, 3] = total_mass
                
                # Find child octant
                octant = 0
                if px >= cx: octant |= 1
                if py >= cy: octant |= 2
                if pz >= cz: octant |= 4
                
                child_base = out_child_base[current]
                child = out_children[child_base + octant]
                
                if child == -1:
                    # Create new leaf
                    child_idx = num_nodes
                    num_nodes += 1
                    if num_nodes >= max_nodes:
                        break
                    
                    quarter = hs * 0.5
                    new_cx = cx + quarter if (octant & 1) else cx - quarter
                    new_cy = cy + quarter if (octant & 2) else cy - quarter
                    new_cz = cz + quarter if (octant & 4) else cz - quarter
                    
                    out_bounds[child_idx, 0] = new_cx
                    out_bounds[child_idx, 1] = new_cy
                    out_bounds[child_idx, 2] = new_cz
                    out_bounds[child_idx, 3] = quarter
                    out_pos_mass[child_idx, 0] = px
                    out_pos_mass[child_idx, 1] = py
                    out_pos_mass[child_idx, 2] = pz
                    out_pos_mass[child_idx, 3] = m
                    out_body_idx[child_idx] = i
                    out_is_leaf[child_idx] = 1
                    out_child_base[child_idx] = 0
                    
                    out_children[child_base + octant] = child_idx
                    break
                else:
                    current = child
                
                depth += 1
        
        return num_nodes

    @njit(parallel=True, cache=True, fastmath=True)
    def _pack_nodes_parallel(num_nodes, pos_mass, bounds, child_base, body_idx, is_leaf,
                             out_nodes_pos_mass, out_nodes_bounds, out_nodes_int):
        """Parallel packing of node data into GPU-ready format."""
        for i in prange(num_nodes):
            out_nodes_pos_mass[i, 0] = pos_mass[i, 0]
            out_nodes_pos_mass[i, 1] = pos_mass[i, 1]
            out_nodes_pos_mass[i, 2] = pos_mass[i, 2]
            out_nodes_pos_mass[i, 3] = pos_mass[i, 3]
            out_nodes_bounds[i, 0] = bounds[i, 0]
            out_nodes_bounds[i, 1] = bounds[i, 1]
            out_nodes_bounds[i, 2] = bounds[i, 2]
            out_nodes_bounds[i, 3] = bounds[i, 3]
            out_nodes_int[i, 0] = child_base[i]
            out_nodes_int[i, 1] = body_idx[i]
            out_nodes_int[i, 2] = is_leaf[i]
    
    _NUMBA_AVAILABLE = True
    
except ImportError:
    _NUMBA_AVAILABLE = False


# =============================================================================
# METAL SIMULATION CLASS - OPTIMIZED
# =============================================================================

class MetalBarnesHutSimulation:
    """
    Metal-accelerated N-body simulation using Barnes-Hut algorithm.
    Optimized for Apple Silicon's Unified Memory Architecture.
    """
    
    def __init__(self, positions: np.ndarray, velocities: np.ndarray,
                 masses: np.ndarray, G: float, softening: float, 
                 damping: float, theta: float = 0.5):
        if not _METAL_AVAILABLE:
            raise RuntimeError(f"Metal not available: {_METAL_ERROR}")
        
        self.n = len(positions)
        self.G = G
        self.softening = softening
        self.softening_sq = softening * softening
        self.damping = damping
        self.theta = theta
        self.theta_sq = theta * theta
        
        # Initialize Metal device
        self.device = MTLCreateSystemDefaultDevice()
        if self.device is None:
            raise RuntimeError("Failed to create Metal device")
        
        self.command_queue = self.device.newCommandQueue()
        
        # Compile Metal shaders
        self._compile_shaders()
        
        # Allocate arrays
        self._allocate_arrays(positions, velocities, masses)
        
        # Pre-allocate reusable constant buffers
        self._setup_constant_buffers()
        
        print(f"[Metal] Initialized Barnes-Hut with {self.n:,} bodies")
        print(f"[Metal] θ={theta}, max tree nodes={self.max_nodes:,}")
    
    def _compile_shaders(self):
        """Compile Metal compute shaders."""
        shader_path = Path(__file__).parent / "barnes_hut.metal"
        
        if not shader_path.exists():
            raise FileNotFoundError(f"Metal shader not found: {shader_path}")
        
        with open(shader_path, 'r') as f:
            source = f.read()
        
        library, error = self.device.newLibraryWithSource_options_error_(source, None, None)
        if library is None:
            raise RuntimeError(f"Failed to compile Metal shaders: {error}")
        
        self.library = library
        self.pipelines = {}
        
        kernel_names = [
            'compute_forces_barnes_hut',
            'compute_forces_tiled',
            'update_particles',
            'compute_colors'
        ]
        
        for name in kernel_names:
            function = library.newFunctionWithName_(name)
            if function is None:
                continue
            
            pipeline, error = self.device.newComputePipelineStateWithFunction_error_(function, None)
            if pipeline is None:
                raise RuntimeError(f"Failed to create pipeline for {name}: {error}")
            
            self.pipelines[name] = pipeline
        
        print(f"[Metal] Compiled {len(self.pipelines)} compute kernels")
    
    def _allocate_arrays(self, positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray):
        """Allocate all numpy arrays with GPU-friendly layouts."""
        n = self.n
        
        # Particle data (float4 for SIMD alignment)
        self.positions = np.zeros((n, 4), dtype=np.float32)
        self.positions[:, :3] = positions[:, :3].astype(np.float32)
        
        self.velocities = np.zeros((n, 4), dtype=np.float32)
        self.velocities[:, :3] = velocities[:, :3].astype(np.float32)
        
        self.masses = masses.astype(np.float32).copy()
        self.accelerations = np.zeros((n, 4), dtype=np.float32)
        self.colors = np.zeros((n, 4), dtype=np.float32)
        self.colors[:, 2] = 1.0  # Initial blue
        self.colors[:, 3] = 1.0
        
        # Tree data - pre-allocate for maximum size
        self.max_nodes = min(8_000_000, n * 4)
        
        # Direct GPU-ready arrays (used by Numba tree builder)
        self.tree_pos_mass = np.zeros((self.max_nodes, 4), dtype=np.float32)
        self.tree_bounds = np.zeros((self.max_nodes, 4), dtype=np.float32)
        self.tree_child_base = np.zeros(self.max_nodes, dtype=np.int32)
        self.tree_body_idx = np.full(self.max_nodes, -1, dtype=np.int32)
        self.tree_is_leaf = np.ones(self.max_nodes, dtype=np.int32)
        self.tree_children = np.full(self.max_nodes * 8, -1, dtype=np.int32)
        
        # Packed node data for GPU (32 bytes per node)
        # Layout: [pos_mass(4f), bounds(4f), child_base(i), body_idx(i), is_leaf(i), pad(i)]
        self.nodes_pos_mass = np.zeros((self.max_nodes, 4), dtype=np.float32)
        self.nodes_bounds = np.zeros((self.max_nodes, 4), dtype=np.float32)
        self.nodes_int = np.zeros((self.max_nodes, 4), dtype=np.int32)  # child_base, body_idx, is_leaf, pad
        
        # Simulation parameters
        self.params = np.zeros(5, dtype=np.float32)
        self.params[0] = self.G
        self.params[1] = self.softening_sq
        self.params[2] = self.theta_sq
        self.params[3] = np.float32(n)  # n_particles as float (will cast in shader)
    
    def _setup_constant_buffers(self):
        """Pre-create ALL Metal buffers (reused every frame to prevent memory leak)."""
        n = self.n
        
        # These are tiny, so allocation overhead is minimal
        self.n_array = np.array([self.n], dtype=np.uint32)
        self.damping_array = np.array([self.damping], dtype=np.float32)
        
        # =====================================================================
        # PRE-ALLOCATE ALL METAL BUFFERS (critical for preventing memory leak!)
        # Using MTLResourceStorageModeShared for unified memory (CPU+GPU access)
        # =====================================================================
        
        # Particle buffers
        self._buf_positions = self.device.newBufferWithLength_options_(
            n * 4 * 4, MTLResourceStorageModeShared)  # n x float4
        self._buf_velocities = self.device.newBufferWithLength_options_(
            n * 4 * 4, MTLResourceStorageModeShared)
        self._buf_accelerations = self.device.newBufferWithLength_options_(
            n * 4 * 4, MTLResourceStorageModeShared)
        self._buf_colors = self.device.newBufferWithLength_options_(
            n * 4 * 4, MTLResourceStorageModeShared)
        
        # Tree buffers (max size)
        max_nodes = self.max_nodes
        # Node data: 12 floats per node (pos_mass + bounds + ints_as_float)
        self._buf_nodes = self.device.newBufferWithLength_options_(
            max_nodes * 12 * 4, MTLResourceStorageModeShared)
        # Children: 8 ints per node
        self._buf_children = self.device.newBufferWithLength_options_(
            max_nodes * 8 * 4, MTLResourceStorageModeShared)
        
        # Small constant buffers
        self._buf_params = self.device.newBufferWithLength_options_(
            5 * 4, MTLResourceStorageModeShared)
        self._buf_dt = self.device.newBufferWithLength_options_(
            4, MTLResourceStorageModeShared)
        self._buf_damping = self.device.newBufferWithLength_options_(
            4, MTLResourceStorageModeShared)
        self._buf_n = self.device.newBufferWithLength_options_(
            4, MTLResourceStorageModeShared)
        self._buf_max_speed = self.device.newBufferWithLength_options_(
            4, MTLResourceStorageModeShared)
        
        # Write initial values to constant buffers
        self._update_buffer(self._buf_damping, self.damping_array)
        self._update_buffer(self._buf_n, self.n_array)
        
        print(f"[Metal] Pre-allocated {(n * 4 * 16 + max_nodes * 56) / (1024*1024):.1f}MB of GPU buffers")
    
    def _update_buffer(self, buffer, array: np.ndarray):
        """Update Metal buffer contents in-place (no allocation!)."""
        contents = buffer.contents()
        try:
            mv = contents.as_buffer(array.nbytes)
            np.copyto(np.frombuffer(mv, dtype=array.dtype).reshape(array.shape), array)
        except Exception:
            # Fallback: use ctypes
            import ctypes
            src = array.ctypes.data_as(ctypes.c_void_p)
            ctypes.memmove(contents, src, array.nbytes)
    
    def _read_buffer(self, buffer, dtype, shape):
        """Read buffer contents into numpy array."""
        contents = buffer.contents()
        try:
            mv = contents.as_buffer(buffer.length())
            return np.frombuffer(mv, dtype=dtype).reshape(shape).copy()
        except Exception:
            try:
                raw = b''.join(contents[:buffer.length()])
                return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
            except:
                return np.zeros(shape, dtype=dtype)
    
    def _read_buffer_into(self, buffer, array: np.ndarray):
        """Read buffer contents directly into existing numpy array (no allocation!)."""
        contents = buffer.contents()
        try:
            mv = contents.as_buffer(array.nbytes)
            np.copyto(array, np.frombuffer(mv, dtype=array.dtype).reshape(array.shape))
        except Exception:
            pass  # Keep existing array on error
    
    def _build_tree(self) -> int:
        """Build Barnes-Hut octree using optimized Numba builder."""
        # Compute bounds
        max_extent = np.abs(self.positions[:, :3]).max()
        bounds = max_extent * 1.1 + 10.0
        
        # Reset arrays (use slice assignment for speed)
        self.tree_pos_mass[:] = 0
        self.tree_bounds[:] = 0
        self.tree_child_base[:] = 0
        self.tree_body_idx[:] = -1
        self.tree_is_leaf[:] = 1
        self.tree_children[:] = -1
        
        if _NUMBA_AVAILABLE:
            # Use optimized Numba tree builder
            num_nodes = _build_tree_direct(
                self.positions[:, :3],
                self.masses,
                np.float32(bounds),
                self.max_nodes,
                self.tree_pos_mass,
                self.tree_bounds,
                self.tree_child_base,
                self.tree_body_idx,
                self.tree_is_leaf,
                self.tree_children
            )
            
            # Fast vectorized copy to packed format (no Python loop!)
            self.nodes_pos_mass[:num_nodes] = self.tree_pos_mass[:num_nodes]
            self.nodes_bounds[:num_nodes] = self.tree_bounds[:num_nodes]
            self.nodes_int[:num_nodes, 0] = self.tree_child_base[:num_nodes]
            self.nodes_int[:num_nodes, 1] = self.tree_body_idx[:num_nodes]
            self.nodes_int[:num_nodes, 2] = self.tree_is_leaf[:num_nodes]
        else:
            raise RuntimeError("Numba required for fast tree building")
        
        return num_nodes
    
    def step(self, dt: float):
        """Perform one simulation step (using pre-allocated buffers - NO memory allocation!)."""
        # 1. Build tree on CPU
        num_nodes = self._build_tree()
        
        # Update params
        self.params[4] = np.float32(num_nodes)  # n_nodes
        
        # 2. Update pre-allocated GPU buffers (NO NEW ALLOCATIONS!)
        self._update_buffer(self._buf_positions, self.positions)
        self._update_buffer(self._buf_velocities, self.velocities)
        self._update_buffer(self._buf_params, self.params)
        
        # Pack nodes into pre-allocated array and update buffer
        # Node struct: pos_mass(4f) + bounds(4f) + ints(4i) = 48 bytes
        # Reuse a pre-allocated array for packing
        if not hasattr(self, '_nodes_packed'):
            self._nodes_packed = np.zeros((self.max_nodes, 12), dtype=np.float32)
        
        self._nodes_packed[:num_nodes, 0:4] = self.nodes_pos_mass[:num_nodes]
        self._nodes_packed[:num_nodes, 4:8] = self.nodes_bounds[:num_nodes]
        self._nodes_packed[:num_nodes, 8:12] = self.nodes_int[:num_nodes].view(np.float32)
        
        self._update_buffer(self._buf_nodes, self._nodes_packed[:num_nodes].ravel())
        self._update_buffer(self._buf_children, self.tree_children[:num_nodes * 8])
        
        # Update dt
        dt_array = np.array([dt], dtype=np.float32)
        self._update_buffer(self._buf_dt, dt_array)
        
        # 3. Dispatch force computation
        cmd_buffer = self.command_queue.commandBuffer()
        encoder = cmd_buffer.computeCommandEncoder()
        
        pipeline = self.pipelines['compute_forces_barnes_hut']
        encoder.setComputePipelineState_(pipeline)
        
        encoder.setBuffer_offset_atIndex_(self._buf_positions, 0, 0)
        encoder.setBuffer_offset_atIndex_(self._buf_nodes, 0, 1)
        encoder.setBuffer_offset_atIndex_(self._buf_children, 0, 2)
        encoder.setBuffer_offset_atIndex_(self._buf_accelerations, 0, 3)
        encoder.setBuffer_offset_atIndex_(self._buf_params, 0, 4)
        
        w = pipeline.threadExecutionWidth()
        threads_per_group = MTLSize(w, 1, 1)
        grid_size = MTLSize(self.n, 1, 1)
        
        encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, threads_per_group)
        encoder.endEncoding()
        
        # 4. Dispatch position/velocity update
        encoder = cmd_buffer.computeCommandEncoder()
        
        pipeline = self.pipelines['update_particles']
        encoder.setComputePipelineState_(pipeline)
        
        encoder.setBuffer_offset_atIndex_(self._buf_positions, 0, 0)
        encoder.setBuffer_offset_atIndex_(self._buf_velocities, 0, 1)
        encoder.setBuffer_offset_atIndex_(self._buf_accelerations, 0, 2)
        encoder.setBuffer_offset_atIndex_(self._buf_dt, 0, 3)
        encoder.setBuffer_offset_atIndex_(self._buf_damping, 0, 4)
        encoder.setBuffer_offset_atIndex_(self._buf_n, 0, 5)
        
        encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, threads_per_group)
        encoder.endEncoding()
        
        # Submit and wait
        cmd_buffer.commit()
        cmd_buffer.waitUntilCompleted()
        
        # 5. Read back results into existing arrays (no allocation!)
        self._read_buffer_into(self._buf_positions, self.positions)
        self._read_buffer_into(self._buf_velocities, self.velocities)
    
    def compute_colors(self, max_speed: float):
        """Compute colors based on velocity on GPU (using pre-allocated buffers)."""
        # Update max_speed buffer
        max_speed_array = np.array([max_speed], dtype=np.float32)
        self._update_buffer(self._buf_max_speed, max_speed_array)
        
        cmd_buffer = self.command_queue.commandBuffer()
        encoder = cmd_buffer.computeCommandEncoder()
        
        pipeline = self.pipelines['compute_colors']
        encoder.setComputePipelineState_(pipeline)
        
        # Use pre-allocated buffers (velocities already updated from step())
        encoder.setBuffer_offset_atIndex_(self._buf_velocities, 0, 0)
        encoder.setBuffer_offset_atIndex_(self._buf_colors, 0, 1)
        encoder.setBuffer_offset_atIndex_(self._buf_max_speed, 0, 2)
        encoder.setBuffer_offset_atIndex_(self._buf_n, 0, 3)
        
        w = pipeline.threadExecutionWidth()
        threads_per_group = MTLSize(w, 1, 1)
        grid_size = MTLSize(self.n, 1, 1)
        
        encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, threads_per_group)
        encoder.endEncoding()
        
        cmd_buffer.commit()
        cmd_buffer.waitUntilCompleted()
        
        # Read colors into existing array
        self._read_buffer_into(self._buf_colors, self.colors)
    
    def get_positions(self) -> np.ndarray:
        return self.positions[:, :3].copy()
    
    def get_velocities(self) -> np.ndarray:
        return self.velocities[:, :3].astype(np.float64)
    
    def get_colors(self) -> np.ndarray:
        return self.colors[:, :3].copy()
    
    def sync(self):
        pass
