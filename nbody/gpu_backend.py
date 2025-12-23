"""
GPU Backend Detection and Selection
====================================

Automatically detects and uses the best available compute backend:
1. CUDA (NVIDIA GPUs) - via Numba CUDA
2. Metal Barnes-Hut (Apple Silicon) - via native Metal compute shaders
3. Metal/MPS (Apple Silicon) - via PyTorch MPS (fallback brute-force)
4. CPU (fallback) - via Numba parallel Barnes-Hut

For N-body simulation:
- CUDA: Tiled brute-force O(n²) for up to 100K bodies
- Metal Barnes-Hut: O(n log n) for ANY body count (leverages UMA)
- Metal MPS: Brute-force O(n²) for small counts only
- CPU: Barnes-Hut O(n log n) as fallback
"""

import os
import sys
import numpy as np
from enum import Enum
from typing import Optional, Tuple

# Suppress warnings during detection
import warnings
warnings.filterwarnings('ignore')


class Backend(Enum):
    CUDA = "cuda"
    METAL_BH = "metal_barnes_hut"  # Native Metal with Barnes-Hut (best for Apple Silicon)
    METAL = "metal"                # PyTorch MPS brute-force (fallback)
    CPU = "cpu"


def detect_backend() -> Tuple[Backend, str]:
    """Detect the best available compute backend."""
    
    # Try CUDA first (NVIDIA GPUs)
    cuda_available, cuda_info = _check_cuda()
    if cuda_available:
        return Backend.CUDA, cuda_info
    
    # Try native Metal Barnes-Hut (best for Apple Silicon)
    metal_bh_available, metal_bh_info = _check_metal_barnes_hut()
    if metal_bh_available:
        return Backend.METAL_BH, metal_bh_info
    
    # Try Metal/MPS via PyTorch (fallback for Apple Silicon)
    metal_available, metal_info = _check_metal()
    if metal_available:
        return Backend.METAL, metal_info
    
    # Fallback to CPU
    return Backend.CPU, _get_cpu_info()


def _check_cuda() -> Tuple[bool, str]:
    """Check if CUDA is available via Numba."""
    try:
        from numba import cuda
        if cuda.is_available():
            device = cuda.get_current_device()
            name = device.name.decode() if isinstance(device.name, bytes) else device.name
            cc = device.compute_capability
            mem = device.total_memory // (1024**3)
            return True, f"{name} (CC {cc[0]}.{cc[1]}, {mem}GB)"
    except Exception:
        pass
    return False, ""


def _check_metal_barnes_hut() -> Tuple[bool, str]:
    """Check if native Metal Barnes-Hut backend is available."""
    try:
        from .metal import is_metal_available
        available, info = is_metal_available()
        if available:
            return True, f"{info} (Barnes-Hut)"
    except ImportError as e:
        # Metal package not installed
        pass
    except Exception as e:
        # Other errors
        pass
    return False, ""


def _check_metal() -> Tuple[bool, str]:
    """Check if Metal/MPS is available via PyTorch."""
    try:
        import torch
        if torch.backends.mps.is_available():
            # Get Apple Silicon chip info
            import platform
            chip = platform.processor() or "Apple Silicon"
            return True, f"Apple {chip} (MPS brute-force)"
    except Exception:
        pass
    return False, ""


def _get_cpu_info() -> str:
    """Get CPU info for fallback."""
    import platform
    try:
        import multiprocessing
        cores = multiprocessing.cpu_count()
        return f"{platform.processor()} ({cores} cores)"
    except Exception:
        return platform.processor() or "Unknown CPU"


# Global backend state
_BACKEND: Optional[Backend] = None
_BACKEND_INFO: str = ""


def get_backend() -> Tuple[Backend, str]:
    """Get the current backend (cached)."""
    global _BACKEND, _BACKEND_INFO
    if _BACKEND is None:
        _BACKEND, _BACKEND_INFO = detect_backend()
        print(f"[GPU] Using backend: {_BACKEND.value} - {_BACKEND_INFO}")
    return _BACKEND, _BACKEND_INFO


def force_backend(backend: Backend):
    """Force a specific backend (for testing)."""
    global _BACKEND, _BACKEND_INFO
    _BACKEND = backend
    _BACKEND_INFO = f"Forced: {backend.value}"


# =============================================================================
# CUDA IMPLEMENTATION (NVIDIA)
# =============================================================================

def _init_cuda_kernels():
    """Initialize CUDA kernels for N-body simulation."""
    from numba import cuda
    import math
    
    # Brute-force kernel - good for <50K bodies on powerful GPUs
    @cuda.jit(fastmath=True)
    def compute_forces_brute_cuda(positions, masses, accelerations, G, softening, n):
        """Compute gravitational forces using brute-force O(n²) on GPU."""
        i = cuda.grid(1)
        if i >= n:
            return
        
        ax, ay, az = 0.0, 0.0, 0.0
        px, py, pz = positions[i, 0], positions[i, 1], positions[i, 2]
        
        for j in range(n):
            if i == j:
                continue
            
            dx = positions[j, 0] - px
            dy = positions[j, 1] - py
            dz = positions[j, 2] - pz
            
            dist_sq = dx*dx + dy*dy + dz*dz + softening*softening
            inv_dist = 1.0 / math.sqrt(dist_sq)
            inv_dist3 = inv_dist * inv_dist * inv_dist
            
            force = G * masses[j] * inv_dist3
            ax += force * dx
            ay += force * dy
            az += force * dz
        
        accelerations[i, 0] = ax
        accelerations[i, 1] = ay
        accelerations[i, 2] = az
    
    # Tiled kernel - uses shared memory for better performance
    TILE_SIZE = 256
    
    @cuda.jit(fastmath=True)
    def compute_forces_tiled_cuda(positions, masses, accelerations, G, softening, n):
        """Tiled force computation with shared memory."""
        # Shared memory for tile
        tile_pos = cuda.shared.array((TILE_SIZE, 3), dtype=np.float64)
        tile_mass = cuda.shared.array(TILE_SIZE, dtype=np.float64)
        
        i = cuda.grid(1)
        tx = cuda.threadIdx.x
        
        ax, ay, az = 0.0, 0.0, 0.0
        
        if i < n:
            px, py, pz = positions[i, 0], positions[i, 1], positions[i, 2]
        else:
            px, py, pz = 0.0, 0.0, 0.0
        
        # Process tiles
        num_tiles = (n + TILE_SIZE - 1) // TILE_SIZE
        
        for tile in range(num_tiles):
            # Load tile into shared memory
            j = tile * TILE_SIZE + tx
            if j < n:
                tile_pos[tx, 0] = positions[j, 0]
                tile_pos[tx, 1] = positions[j, 1]
                tile_pos[tx, 2] = positions[j, 2]
                tile_mass[tx] = masses[j]
            else:
                tile_pos[tx, 0] = 0.0
                tile_pos[tx, 1] = 0.0
                tile_pos[tx, 2] = 0.0
                tile_mass[tx] = 0.0
            
            cuda.syncthreads()
            
            # Compute forces from this tile
            if i < n:
                for k in range(TILE_SIZE):
                    j = tile * TILE_SIZE + k
                    if j >= n or j == i:
                        continue
                    
                    dx = tile_pos[k, 0] - px
                    dy = tile_pos[k, 1] - py
                    dz = tile_pos[k, 2] - pz
                    
                    dist_sq = dx*dx + dy*dy + dz*dz + softening*softening
                    inv_dist = 1.0 / math.sqrt(dist_sq)
                    inv_dist3 = inv_dist * inv_dist * inv_dist
                    
                    force = G * tile_mass[k] * inv_dist3
                    ax += force * dx
                    ay += force * dy
                    az += force * dz
            
            cuda.syncthreads()
        
        if i < n:
            accelerations[i, 0] = ax
            accelerations[i, 1] = ay
            accelerations[i, 2] = az
    
    @cuda.jit(fastmath=True)
    def update_bodies_cuda(positions, velocities, accelerations, dt, damping, n):
        """Update positions and velocities (leapfrog integration)."""
        i = cuda.grid(1)
        if i >= n:
            return
        
        # Update velocity (half step already done, complete it)
        velocities[i, 0] = (velocities[i, 0] + accelerations[i, 0] * dt) * damping
        velocities[i, 1] = (velocities[i, 1] + accelerations[i, 1] * dt) * damping
        velocities[i, 2] = (velocities[i, 2] + accelerations[i, 2] * dt) * damping
        
        # Update position
        positions[i, 0] += velocities[i, 0] * dt
        positions[i, 1] += velocities[i, 1] * dt
        positions[i, 2] += velocities[i, 2] * dt
    
    @cuda.jit(fastmath=True)
    def compute_colors_cuda(velocities, colors, n, max_speed):
        """Compute colors based on velocity (deep blue → red heat map)."""
        i = cuda.grid(1)
        if i >= n:
            return
        
        vx, vy, vz = velocities[i, 0], velocities[i, 1], velocities[i, 2]
        speed = math.sqrt(vx*vx + vy*vy + vz*vz)
        t = min(speed / max_speed, 1.0)
        
        # Color gradient: deep blue → light blue → cyan → white → yellow → orange → red
        if t < 0.2:
            # Deep blue → Light blue
            s = t * 5.0
            colors[i, 0] = 0.0 + 0.3 * s
            colors[i, 1] = 0.1 + 0.4 * s
            colors[i, 2] = 0.5 + 0.4 * s
        elif t < 0.4:
            # Light blue → Cyan
            s = (t - 0.2) * 5.0
            colors[i, 0] = 0.3 - 0.1 * s
            colors[i, 1] = 0.5 + 0.3 * s
            colors[i, 2] = 0.9 + 0.1 * s
        elif t < 0.6:
            # Cyan → White
            s = (t - 0.4) * 5.0
            colors[i, 0] = 0.2 + 0.8 * s
            colors[i, 1] = 0.8 + 0.2 * s
            colors[i, 2] = 1.0
        elif t < 0.8:
            # White → Yellow
            s = (t - 0.6) * 5.0
            colors[i, 0] = 1.0
            colors[i, 1] = 1.0 - 0.05 * s
            colors[i, 2] = 1.0 - 1.0 * s
        elif t < 0.9:
            # Yellow → Orange (rare)
            s = (t - 0.8) * 10.0
            colors[i, 0] = 1.0
            colors[i, 1] = 0.95 - 0.45 * s
            colors[i, 2] = 0.0
        else:
            # Orange → Red (extremely rare!)
            s = (t - 0.9) * 10.0
            colors[i, 0] = 1.0
            colors[i, 1] = 0.5 - 0.5 * s
            colors[i, 2] = 0.0
    
    return {
        'brute': compute_forces_brute_cuda,
        'tiled': compute_forces_tiled_cuda,
        'update': update_bodies_cuda,
        'colors': compute_colors_cuda,
        'tile_size': TILE_SIZE,
    }


class CUDASimulation:
    """CUDA-accelerated N-body simulation."""
    
    def __init__(self, positions: np.ndarray, velocities: np.ndarray, 
                 masses: np.ndarray, G: float, softening: float, damping: float):
        from numba import cuda
        
        self.n = len(positions)
        self.G = G
        self.softening = softening
        self.damping = damping
        
        # Initialize kernels
        self.kernels = _init_cuda_kernels()
        
        # Allocate device arrays
        self.d_positions = cuda.to_device(positions.astype(np.float64))
        self.d_velocities = cuda.to_device(velocities.astype(np.float64))
        self.d_masses = cuda.to_device(masses.astype(np.float64))
        self.d_accelerations = cuda.device_array((self.n, 3), dtype=np.float64)
        self.d_colors = cuda.device_array((self.n, 3), dtype=np.float32)
        
        # Grid/block configuration
        self.threads_per_block = 256
        self.blocks = (self.n + self.threads_per_block - 1) // self.threads_per_block
        
        # Choose force kernel based on body count
        self.use_tiled = self.n > 10000
        
        print(f"[CUDA] Initialized with {self.n:,} bodies")
        print(f"[CUDA] Using {'tiled' if self.use_tiled else 'brute-force'} kernel")
    
    def step(self, dt: float):
        """Perform one simulation step."""
        # Compute forces
        if self.use_tiled:
            self.kernels['tiled'][self.blocks, self.threads_per_block](
                self.d_positions, self.d_masses, self.d_accelerations,
                self.G, self.softening, self.n
            )
        else:
            self.kernels['brute'][self.blocks, self.threads_per_block](
                self.d_positions, self.d_masses, self.d_accelerations,
                self.G, self.softening, self.n
            )
        
        # Update positions/velocities
        self.kernels['update'][self.blocks, self.threads_per_block](
            self.d_positions, self.d_velocities, self.d_accelerations,
            dt, self.damping, self.n
        )
    
    def compute_colors(self, max_speed: float):
        """Compute colors on GPU."""
        self.kernels['colors'][self.blocks, self.threads_per_block](
            self.d_velocities, self.d_colors, self.n, max_speed
        )
    
    def get_positions(self) -> np.ndarray:
        """Copy positions back to host."""
        return self.d_positions.copy_to_host().astype(np.float32)
    
    def get_velocities(self) -> np.ndarray:
        """Copy velocities back to host."""
        return self.d_velocities.copy_to_host().astype(np.float64)
    
    def get_colors(self) -> np.ndarray:
        """Copy colors back to host."""
        return self.d_colors.copy_to_host()
    
    def sync(self):
        """Synchronize GPU."""
        from numba import cuda
        cuda.synchronize()


# =============================================================================
# METAL/MPS IMPLEMENTATION (Apple Silicon)
# =============================================================================

class MetalSimulation:
    """Metal/MPS-accelerated N-body simulation via PyTorch.
    
    Uses tiled/blocked computation to handle large body counts efficiently
    while staying within GPU memory limits.
    """
    
    def __init__(self, positions: np.ndarray, velocities: np.ndarray,
                 masses: np.ndarray, G: float, softening: float, damping: float):
        import torch
        
        self.device = torch.device("mps")
        self.n = len(positions)
        self.G = G
        self.softening = softening
        self.softening_sq = softening * softening
        self.damping = damping
        
        # Move to GPU
        self.positions = torch.from_numpy(positions.astype(np.float32)).to(self.device)
        self.velocities = torch.from_numpy(velocities.astype(np.float32)).to(self.device)
        self.masses = torch.from_numpy(masses.astype(np.float32)).to(self.device)
        self.accelerations = torch.zeros((self.n, 3), dtype=torch.float32, device=self.device)
        self.colors = torch.zeros((self.n, 3), dtype=torch.float32, device=self.device)
        
        # Tile size - tuned for MPS memory and compute
        # Larger tiles = faster but more memory
        self.tile_size = min(2048, self.n)
        
        # Pre-compute tile indices
        self.n_tiles = (self.n + self.tile_size - 1) // self.tile_size
        
        print(f"[Metal] Initialized with {self.n:,} bodies on MPS (tiles: {self.n_tiles})")
    
    def step(self, dt: float):
        """Perform one simulation step using optimized PyTorch operations."""
        import torch
        
        # Reset accelerations
        self.accelerations.zero_()
        
        # Tiled force computation - O(n²) but GPU-parallelized
        for ti in range(self.n_tiles):
            i_start = ti * self.tile_size
            i_end = min(i_start + self.tile_size, self.n)
            
            # Get positions for this tile
            pos_i = self.positions[i_start:i_end]  # (tile_i, 3)
            tile_i_size = i_end - i_start
            
            # Accumulate forces from all other tiles
            acc_i = torch.zeros((tile_i_size, 3), dtype=torch.float32, device=self.device)
            
            for tj in range(self.n_tiles):
                j_start = tj * self.tile_size
                j_end = min(j_start + self.tile_size, self.n)
                
                pos_j = self.positions[j_start:j_end]  # (tile_j, 3)
                mass_j = self.masses[j_start:j_end]    # (tile_j,)
                tile_j_size = j_end - j_start
                
                # Compute displacement vectors: (tile_i, tile_j, 3)
                diff = pos_j.unsqueeze(0) - pos_i.unsqueeze(1)
                
                # Squared distances: (tile_i, tile_j)
                dist_sq = (diff * diff).sum(dim=2) + self.softening_sq
                
                # Handle self-interaction (set to inf to zero out force)
                if ti == tj:
                    # Diagonal elements of this tile-tile interaction
                    diag_idx = torch.arange(min(tile_i_size, tile_j_size), device=self.device)
                    dist_sq[diag_idx, diag_idx] = float('inf')
                
                # Inverse distance cubed: (tile_i, tile_j)
                inv_dist = torch.rsqrt(dist_sq)
                inv_dist3 = inv_dist * inv_dist * inv_dist
                
                # Force contribution: (tile_i, tile_j, 3)
                # F = G * m_j / r³ * (r_j - r_i)
                force_factor = self.G * mass_j.unsqueeze(0) * inv_dist3  # (tile_i, tile_j)
                force = force_factor.unsqueeze(2) * diff  # (tile_i, tile_j, 3)
                
                # Sum over j: (tile_i, 3)
                acc_i += force.sum(dim=1)
            
            self.accelerations[i_start:i_end] = acc_i
        
        # Leapfrog integration
        self.velocities = (self.velocities + self.accelerations * dt) * self.damping
        self.positions = self.positions + self.velocities * dt
    
    def compute_colors(self, max_speed: float):
        """Compute colors based on velocity (deep blue → red heat map)."""
        import torch
        speed = torch.norm(self.velocities, dim=1)
        t = torch.clamp(speed / max_speed, 0, 1)
        
        # Color gradient: deep blue → light blue → cyan → white → yellow → orange → red
        # Initialize colors
        r = torch.zeros_like(t)
        g = torch.zeros_like(t)
        b = torch.zeros_like(t)
        
        # Deep blue → Light blue (0.0-0.2)
        mask = t < 0.2
        s = t[mask] * 5.0
        r[mask] = 0.0 + 0.3 * s
        g[mask] = 0.1 + 0.4 * s
        b[mask] = 0.5 + 0.4 * s
        
        # Light blue → Cyan (0.2-0.4)
        mask = (t >= 0.2) & (t < 0.4)
        s = (t[mask] - 0.2) * 5.0
        r[mask] = 0.3 - 0.1 * s
        g[mask] = 0.5 + 0.3 * s
        b[mask] = 0.9 + 0.1 * s
        
        # Cyan → White (0.4-0.6)
        mask = (t >= 0.4) & (t < 0.6)
        s = (t[mask] - 0.4) * 5.0
        r[mask] = 0.2 + 0.8 * s
        g[mask] = 0.8 + 0.2 * s
        b[mask] = 1.0
        
        # White → Yellow (0.6-0.8)
        mask = (t >= 0.6) & (t < 0.8)
        s = (t[mask] - 0.6) * 5.0
        r[mask] = 1.0
        g[mask] = 1.0 - 0.05 * s
        b[mask] = 1.0 - 1.0 * s
        
        # Yellow → Orange (0.8-0.9, rare)
        mask = (t >= 0.8) & (t < 0.9)
        s = (t[mask] - 0.8) * 10.0
        r[mask] = 1.0
        g[mask] = 0.95 - 0.45 * s
        b[mask] = 0.0
        
        # Orange → Red (0.9-1.0, extremely rare!)
        mask = t >= 0.9
        s = (t[mask] - 0.9) * 10.0
        r[mask] = 1.0
        g[mask] = 0.5 - 0.5 * s
        b[mask] = 0.0
        
        self.colors[:, 0] = r
        self.colors[:, 1] = g
        self.colors[:, 2] = b
    
    def get_positions(self) -> np.ndarray:
        """Copy positions back to CPU."""
        return self.positions.cpu().numpy()
    
    def get_velocities(self) -> np.ndarray:
        """Copy velocities back to CPU."""
        return self.velocities.cpu().numpy().astype(np.float64)
    
    def get_colors(self) -> np.ndarray:
        """Copy colors back to CPU."""
        return self.colors.cpu().numpy()
    
    def sync(self):
        """Synchronize MPS."""
        import torch
        torch.mps.synchronize()


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

# Body count thresholds for GPU vs CPU
# GPU brute-force is O(n²), CPU Barnes-Hut is O(n log n)
# GPU is faster for small n, CPU for large n
CUDA_THRESHOLD = 100_000        # CUDA is fast even for large counts (RTX 4080 can handle 1M+)
METAL_BH_THRESHOLD = 2_000_000  # Metal Barnes-Hut can handle millions (limited by RAM only)
METAL_THRESHOLD = 5_000         # Metal MPS brute-force slower than CPU Barnes-Hut after ~5K


def create_gpu_simulation(positions: np.ndarray, velocities: np.ndarray,
                          masses: np.ndarray, G: float, softening: float, 
                          damping: float, theta: float = 0.5, force_gpu: bool = False):
    """Create the appropriate GPU simulation based on available backend.
    
    Args:
        positions: Initial positions (n, 3)
        velocities: Initial velocities (n, 3)
        masses: Body masses (n,)
        G: Gravitational constant
        softening: Softening length
        damping: Velocity damping
        theta: Barnes-Hut opening angle (for Metal Barnes-Hut)
        force_gpu: If True, use GPU even if CPU would be faster
    
    Returns:
        GPU simulation object or None if CPU is better
    """
    backend, info = get_backend()
    n = len(positions)
    
    if backend == Backend.CUDA:
        # CUDA is fast even for large body counts
        if n <= CUDA_THRESHOLD or force_gpu:
            return CUDASimulation(positions, velocities, masses, G, softening, damping)
        else:
            print(f"[GPU] {n:,} bodies exceeds CUDA threshold, using CPU Barnes-Hut")
            return None
    
    elif backend == Backend.METAL_BH:
        # Metal Barnes-Hut can handle very large counts efficiently
        if n <= METAL_BH_THRESHOLD or force_gpu:
            try:
                from .metal import MetalBarnesHutSimulation
                return MetalBarnesHutSimulation(
                    positions, velocities, masses, G, softening, damping, theta
                )
            except Exception as e:
                print(f"[GPU] Metal Barnes-Hut init failed: {e}")
                # Try fallback to PyTorch MPS
                try:
                    return MetalSimulation(positions, velocities, masses, G, softening, damping)
                except:
                    return None
        else:
            print(f"[GPU] {n:,} bodies exceeds Metal threshold ({METAL_BH_THRESHOLD:,})")
            return None
    
    elif backend == Backend.METAL:
        # Metal MPS brute-force is only efficient for smaller counts
        if n <= METAL_THRESHOLD or force_gpu:
            return MetalSimulation(positions, velocities, masses, G, softening, damping)
        else:
            print(f"[GPU] {n:,} bodies exceeds Metal MPS threshold ({METAL_THRESHOLD:,}), using CPU Barnes-Hut")
            return None
    
    return None  # CPU fallback

