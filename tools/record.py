"""
N-Body Offline Renderer
=======================

Computes simulation frames offline (can take hours) and saves them to disk.
Each frame can take minutes to compute with 1M bodies and low theta.

Usage:
    python -m tools.record                     # Start new recording
    python -m tools.record --resume            # Resume interrupted recording
    python -m tools.record --status            # Check recording status
    python -m tools.record --list              # List all recordings
    python -m tools.record --resume galaxy_1m  # Resume specific session

Output:
    recordings/<session_name>/
        metadata.json     - Recording settings
        frame_0000.npz    - Position/color data for each frame
        ...
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Get project root (parent of tools/)
PROJECT_ROOT = Path(__file__).parent.parent

# =============================================================================
# RECORDING PRESETS - Uncomment the one you want to use
# =============================================================================

# PRESET: FAST & SMOOTH (~1-2 hours, 500K bodies, 2000 frames)
# Visually impressive, fast render, ~100 second video at 20fps
RECORDING_CONFIG = {
    "session_name": "collision_fast_500k-3",
    "num_bodies": 100_000,            # Half million bodies!
    "theta": 0.95,                    # Very fast (approximate physics)
    "G": 0.15,                        # Slightly stronger gravity
    "softening": 3.0,
    "damping": 1.0,
    "spawn_radius": 600.0,
    "distribution": "collision",
    "total_frames": 4000,             # LOTS of frames
    "dt_per_frame": 0.15,
    "substeps": 2,                    # Minimal for speed
    "target_fps": 20,
}

# PRESET: COLLISION (~1.5 hours, 300K bodies, 1500 frames, two galaxies)
# RECORDING_CONFIG = {
#     "session_name": "collision_300k",
#     "num_bodies": 300_000,
#     "theta": 0.9,
#     "G": 0.12,
#     "softening": 2.5,
#     "damping": 1.0,
#     "spawn_radius": 500.0,
#     "distribution": "collision",      # Two galaxies colliding!
#     "total_frames": 1500,
#     "dt_per_frame": 0.2,
#     "substeps": 2,
#     "target_fps": 20,
# }

# PRESET: MEGA FRAMES (~1 hour, 200K bodies, 4000 frames, ultra smooth)
# RECORDING_CONFIG = {
#     "session_name": "galaxy_ultrasmooth",
#     "num_bodies": 200_000,
#     "theta": 0.95,
#     "G": 0.1,
#     "softening": 2.0,
#     "damping": 1.0,
#     "spawn_radius": 500.0,
#     "distribution": "galaxy",
#     "total_frames": 4000,             # 200 seconds at 20fps!
#     "dt_per_frame": 0.1,
#     "substeps": 1,                    # Single step = max speed
#     "target_fps": 20,
# }

# PRESET: SPIRAL GALAXY (~1-2 hours, 400K bodies, Milky Way style)
# RECORDING_CONFIG = {
#     "session_name": "spiral_milkyway",
#     "num_bodies": 400_000,
#     "theta": 0.9,
#     "G": 0.08,                        # Tuned for spiral stability
#     "softening": 2.0,
#     "damping": 1.0,
#     "spawn_radius": 600.0,
#     "distribution": "spiral",         # Spiral arms + central bulge!
#     "total_frames": 1500,
#     "dt_per_frame": 0.12,
#     "substeps": 3,
#     "target_fps": 20,
# }


def get_recording_dir(session_name: str) -> Path:
    """Get the directory for a recording session."""
    base = PROJECT_ROOT / "recordings" / session_name
    base.mkdir(parents=True, exist_ok=True)
    return base


def save_metadata(rec_dir: Path, config: dict, start_time: float):
    """Save recording metadata."""
    metadata = {
        **config,
        "start_time": start_time,
        "start_datetime": datetime.fromtimestamp(start_time).isoformat(),
    }
    with open(rec_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def load_metadata(rec_dir: Path) -> dict:
    """Load recording metadata."""
    with open(rec_dir / "metadata.json", "r") as f:
        return json.load(f)


def get_completed_frames(rec_dir: Path) -> int:
    """Count how many frames have been recorded."""
    count = 0
    while (rec_dir / f"frame_{count:04d}.npz").exists():
        count += 1
    return count


def find_latest_state(rec_dir: Path, max_frame: int) -> tuple:
    """Find the most recent state file and its frame number."""
    for frame in range(max_frame, -1, -1):
        state_file = rec_dir / f"state_{frame:04d}.npz"
        if state_file.exists():
            return state_file, frame
    return None, -1


def save_frame(rec_dir: Path, frame_idx: int, positions: np.ndarray, colors: np.ndarray):
    """Save a single frame to disk (uncompressed for speed)."""
    # Using uncompressed saves: 4ms vs 62ms for compressed
    # Background compression will compress batches later
    np.savez(
        rec_dir / f"frame_{frame_idx:04d}.npz",
        positions=positions.astype(np.float32),
        colors=colors.astype(np.float32),
    )


# =============================================================================
# BACKGROUND BATCH COMPRESSION
# =============================================================================

import threading
from queue import Queue
import shutil

# Compression batch size - compress this many frames at once
# Smaller batches = less memory pressure, more frequent cleanup
COMPRESSION_BATCH_SIZE = 50

import gc  # For explicit garbage collection


class BackgroundCompressor:
    """
    Compresses frames in batches in the background while recording continues.
    
    Strategy:
    - Frames are saved uncompressed for speed (~4ms each)
    - Every BATCH_SIZE frames, queue them for background compression
    - Background thread compresses and replaces uncompressed files
    - Final result: compressed files with fast recording
    """
    
    def __init__(self, rec_dir: Path, batch_size: int = COMPRESSION_BATCH_SIZE):
        self.rec_dir = rec_dir
        self.batch_size = batch_size
        self.queue = Queue()
        self.thread = None
        self.running = False
        self.compressed_count = 0
        self.pending_batches = 0
    
    def start(self):
        """Start the background compression thread."""
        self.running = True
        self.thread = threading.Thread(target=self._compress_worker, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop the background thread and wait for completion."""
        self.running = False
        self.queue.put(None)  # Signal to stop
        if self.thread:
            self.thread.join(timeout=60)  # Wait up to 60s for completion
    
    def queue_batch(self, start_frame: int, end_frame: int):
        """Queue a batch of frames for compression."""
        self.queue.put((start_frame, end_frame))
        self.pending_batches += 1
    
    def check_and_queue(self, current_frame: int):
        """Check if we should queue a new batch for compression."""
        # Queue compression when we've completed a batch
        if (current_frame + 1) % self.batch_size == 0:
            start = current_frame - self.batch_size + 1
            end = current_frame + 1
            self.queue_batch(start, end)
    
    def _compress_worker(self):
        """Background worker that compresses batches."""
        while self.running:
            try:
                item = self.queue.get(timeout=1.0)
                if item is None:
                    break
                
                start_frame, end_frame = item
                self._compress_batch(start_frame, end_frame)
                self.pending_batches -= 1
                
                # Force garbage collection after each batch to free memory
                gc.collect()
                
            except:
                continue
    
    def _compress_batch(self, start_frame: int, end_frame: int):
        """Compress a batch of frames."""
        for frame_idx in range(start_frame, end_frame):
            uncompressed = self.rec_dir / f"frame_{frame_idx:04d}.npz"
            
            if not uncompressed.exists():
                continue
            
            try:
                # Load uncompressed - MUST use context manager to close file!
                with np.load(uncompressed) as data:
                    positions = data['positions'].copy()  # Copy before file closes
                    colors = data['colors'].copy()
                
                # Save compressed (to temp file first)
                temp_file = self.rec_dir / f"frame_{frame_idx:04d}.tmp.npz"
                np.savez_compressed(temp_file, positions=positions, colors=colors)
                
                # Replace original with compressed
                temp_file.replace(uncompressed)
                self.compressed_count += 1
                
                # Explicitly free memory
                del positions, colors
                
            except Exception as e:
                # If compression fails, keep the uncompressed file
                pass
    
    def compress_remaining(self, total_frames: int):
        """Compress any remaining uncompressed frames after recording."""
        # Find the last queued batch end
        last_batch_end = (total_frames // self.batch_size) * self.batch_size
        
        # Compress remaining frames
        if last_batch_end < total_frames:
            print(f"[Compress] Compressing final {total_frames - last_batch_end} frames...")
            self._compress_batch(last_batch_end, total_frames)
        
        # Wait for background thread to finish
        while self.pending_batches > 0:
            import time
            time.sleep(0.1)
        
        print(f"[Compress] Compressed {self.compressed_count}/{total_frames} frames")
    
    def get_status(self) -> str:
        """Get compression status string."""
        if self.pending_batches > 0:
            return f"(compressing: {self.pending_batches} batches pending)"
        return ""


def format_time(seconds: float) -> str:
    """Format seconds as human-readable time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def format_eta(seconds: float) -> str:
    """Format ETA."""
    if seconds < 0:
        return "calculating..."
    td = timedelta(seconds=int(seconds))
    return str(td)


def print_progress(frame: int, total: int, frame_time: float, elapsed: float, eta: float):
    """Print progress for each frame on its own line."""
    pct = (frame + 1) / total * 100
    bar_width = 20
    filled = int(bar_width * (frame + 1) / total)
    bar = "█" * filled + "░" * (bar_width - filled)
    
    print(f"[{bar}] {pct:5.1f}% | Frame {frame+1:4d}/{total} | "
          f"Time: {format_time(frame_time):>6s} | Elapsed: {format_time(elapsed):>6s} | "
          f"ETA: {format_eta(eta)}")


def _generate_initial_conditions(config: dict):
    """Generate initial positions, velocities, masses based on distribution.
    
    This avoids creating a full NBodySimulation which would initialize GPU twice.
    """
    n = config["num_bodies"]
    R = config["spawn_radius"]
    G = config["G"]
    distribution = config.get("distribution", "galaxy")
    
    if distribution == "galaxy":
        # Disk-shaped distribution
        r = np.random.exponential(R * 0.3, n)
        theta = np.random.uniform(0, 2 * np.pi, n)
        z = np.random.normal(0, R * 0.02, n)
        
        positions = np.zeros((n, 3), dtype=np.float64)
        positions[:, 0] = r * np.cos(theta)
        positions[:, 1] = z
        positions[:, 2] = r * np.sin(theta)
        
        velocities = np.zeros((n, 3), dtype=np.float64)
        orbital_speed = np.sqrt(G * n * 0.001 / (r + 1.0))
        velocities[:, 0] = -orbital_speed * np.sin(theta)
        velocities[:, 2] = orbital_speed * np.cos(theta)
        velocities[:, 1] = np.random.normal(0, orbital_speed * 0.1, n)
        
    elif distribution == "collision":
        # Two galaxies colliding
        half = n // 2
        positions = np.zeros((n, 3), dtype=np.float64)
        velocities = np.zeros((n, 3), dtype=np.float64)
        
        # Galaxy 1
        r1 = np.random.exponential(R * 0.2, half)
        theta1 = np.random.uniform(0, 2 * np.pi, half)
        positions[:half, 0] = r1 * np.cos(theta1) - R * 0.4
        positions[:half, 1] = np.random.normal(0, R * 0.02, half)
        positions[:half, 2] = r1 * np.sin(theta1)
        
        orbital_speed1 = np.sqrt(G * half * 0.001 / (r1 + 1.0))
        velocities[:half, 0] = -orbital_speed1 * np.sin(theta1) + 2.0
        velocities[:half, 2] = orbital_speed1 * np.cos(theta1)
        
        # Galaxy 2
        r2 = np.random.exponential(R * 0.2, n - half)
        theta2 = np.random.uniform(0, 2 * np.pi, n - half)
        positions[half:, 0] = r2 * np.cos(theta2) + R * 0.4
        positions[half:, 1] = np.random.normal(0, R * 0.02, n - half)
        positions[half:, 2] = r2 * np.sin(theta2)
        
        orbital_speed2 = np.sqrt(G * (n - half) * 0.001 / (r2 + 1.0))
        velocities[half:, 0] = -orbital_speed2 * np.sin(theta2) - 2.0
        velocities[half:, 2] = orbital_speed2 * np.cos(theta2)
        
    elif distribution == "spiral":
        # Spiral galaxy (simplified)
        positions = np.zeros((n, 3), dtype=np.float64)
        velocities = np.zeros((n, 3), dtype=np.float64)
        
        disk_r = np.random.exponential(R * 0.25, n)
        disk_r = np.clip(disk_r, R * 0.02, R * 0.9)
        
        spiral_tightness = 0.3
        base_theta = np.log(disk_r / (R * 0.05) + 1) / spiral_tightness
        arm_assignment = np.random.randint(0, 4, n)
        arm_offset = arm_assignment * (2 * np.pi / 4)
        theta_scatter = np.random.normal(0, 0.3, n)
        disk_theta = base_theta + arm_offset + theta_scatter
        disk_z = np.random.normal(0, R * 0.01, n) * (1 + disk_r / R)
        
        positions[:, 0] = disk_r * np.cos(disk_theta)
        positions[:, 1] = disk_z
        positions[:, 2] = disk_r * np.sin(disk_theta)
        
        central_mass = n * 50.0
        enclosed_mass = central_mass + disk_r / R * n * 0.5
        orbital_speed = np.sqrt(G * enclosed_mass / (disk_r + 0.1))
        
        velocities[:, 0] = -orbital_speed * np.sin(disk_theta)
        velocities[:, 2] = orbital_speed * np.cos(disk_theta)
        velocities += np.random.normal(0, orbital_speed[:, np.newaxis] * 0.05, (n, 3))
        
    else:  # sphere or uniform
        phi = np.random.uniform(0, 2 * np.pi, n)
        cos_theta = np.random.uniform(-1, 1, n)
        sin_theta = np.sqrt(1 - cos_theta**2)
        r = R * 0.8 * np.cbrt(np.random.uniform(0, 1, n))
        
        positions = np.zeros((n, 3), dtype=np.float64)
        positions[:, 0] = r * sin_theta * np.cos(phi)
        positions[:, 1] = r * sin_theta * np.sin(phi)
        positions[:, 2] = r * cos_theta
        
        velocities = np.random.normal(0, 0.5, (n, 3)).astype(np.float64)
    
    # Masses - mostly uniform with some heavier particles
    masses = np.ones(n, dtype=np.float64)
    heavy_count = max(1, n // 1000)
    heavy_indices = np.random.choice(n, heavy_count, replace=False)
    masses[heavy_indices] = 100.0
    
    return positions, velocities, masses


def record(config: dict, resume: bool = False):
    """Main recording function with GPU acceleration when available."""
    # Import here to avoid loading heavy deps for --status
    from nbody.simulation import (
        build_octree, compute_forces_barnes_hut,
        update_positions_velocities, compute_colors_by_velocity, compute_bounds
    )
    
    rec_dir = get_recording_dir(config["session_name"])
    
    # Check for resume
    start_frame = 0
    positions = None
    velocities = None
    masses = None
    
    if resume:
        completed_frames = get_completed_frames(rec_dir)
        if completed_frames > 0:
            print(f"[Record] Found {completed_frames} completed frames")
            metadata = load_metadata(rec_dir)
            
            state_file, state_frame = find_latest_state(rec_dir, completed_frames)
            
            if state_file is not None:
                print(f"[Record] Loading state from frame {state_frame}")
                with np.load(state_file) as state:
                    positions = state["positions"].copy()
                    velocities = state["velocities"].copy()
                start_frame = state_frame + 1
                print(f"[Record] Resuming from frame {start_frame}")
            else:
                print("[Record] Warning: No state file found, recomputing from start...")
                start_frame = 0
    
    if positions is None:
        print(f"[Record] Starting new recording: {config['session_name']}")
        print(f"[Record] Bodies: {config['num_bodies']:,}, θ={config['theta']}")
        print(f"[Record] Frames: {config['total_frames']}, dt={config['dt_per_frame']}")
        
        # Generate initial conditions directly (avoid creating NBodySimulation which initializes GPU)
        positions, velocities, masses = _generate_initial_conditions(config)
        
        save_metadata(rec_dir, config, time.time())
    
    num_bodies = config["num_bodies"]
    total_frames = config["total_frames"]
    dt = config["dt_per_frame"] / config["substeps"]
    substeps = config["substeps"]
    
    if masses is None:
        masses = np.ones(num_bodies, dtype=np.float64)
    
    # Try GPU acceleration
    gpu_sim = None
    use_gpu = False
    
    try:
        from nbody.gpu_backend import get_backend, Backend, create_gpu_simulation
        backend, info = get_backend()
        
        if backend != Backend.CPU:
            # Pass theta for Barnes-Hut backends (Metal and CPU)
            theta = config.get("theta", 0.5)
            
            # Only force GPU for efficient backends (Barnes-Hut)
            # Don't force brute-force MPS for large body counts
            force_gpu = (backend == Backend.METAL_BH or backend == Backend.CUDA)
            
            gpu_sim = create_gpu_simulation(
                positions, velocities, masses,
                config["G"], config["softening"], config["damping"],
                theta=theta, force_gpu=force_gpu
            )
            if gpu_sim is not None:
                use_gpu = True
                print(f"[Record] GPU acceleration: {backend.value} - {info}")
                if backend == Backend.METAL_BH:
                    print(f"[Record] Using Metal Barnes-Hut with θ={theta} (UMA zero-copy)")
    except Exception as e:
        print(f"[Record] GPU not available: {e}")
        import traceback
        traceback.print_exc()
    
    if not use_gpu:
        print("[Record] Using CPU backend (Barnes-Hut + Numba)")
        # CPU-only arrays
        theta = config["theta"]
        G = config["G"]
        softening = config["softening"]
        damping = config["damping"]
        accelerations = np.zeros((num_bodies, 3), dtype=np.float64)
        
        max_nodes = min(8_000_000, num_bodies * 4)
        node_centers = np.zeros((max_nodes, 3), dtype=np.float64)
        node_half_sizes = np.zeros(max_nodes, dtype=np.float64)
        node_masses = np.zeros(max_nodes, dtype=np.float64)
        node_com = np.zeros((max_nodes, 3), dtype=np.float64)
        node_children = np.full((max_nodes, 8), -1, dtype=np.int32)
        node_body_idx = np.full(max_nodes, -1, dtype=np.int32)
        node_is_leaf = np.ones(max_nodes, dtype=np.bool_)
    
    colors = np.zeros((num_bodies, 3), dtype=np.float32)
    
    # Start background compressor
    compressor = BackgroundCompressor(rec_dir, batch_size=COMPRESSION_BATCH_SIZE)
    compressor.start()
    
    print(f"\n[Record] Starting computation at frame {start_frame}...")
    print(f"[Record] Background compression every {COMPRESSION_BATCH_SIZE} frames")
    print("[Record] Press Ctrl+C to pause (can resume later)\n")
    
    start_time = time.time()
    frame_times = []
    
    try:
        for frame in range(start_frame, total_frames):
            frame_start = time.time()
            
            if use_gpu:
                # GPU path - optimized for speed
                for _ in range(substeps):
                    gpu_sim.step(dt)
                
                gpu_sim.compute_colors(15.0)
                # Keep as float32 - no conversion needed for saving
                positions = gpu_sim.get_positions()
                colors = gpu_sim.get_colors()
                # Only get velocities when we need to save state (every 50 frames)
                if (frame + 1) % 50 == 0:
                    velocities = gpu_sim.get_velocities()
            else:
                # CPU path with Barnes-Hut
                for _ in range(substeps):
                    bounds = compute_bounds(positions, num_bodies)
                    
                    node_children.fill(-1)
                    node_body_idx.fill(-1)
                    node_is_leaf.fill(True)
                    
                    num_nodes = build_octree(
                        positions, masses, num_bodies, bounds,
                        node_centers, node_half_sizes, node_masses, node_com,
                        node_children, node_body_idx, node_is_leaf
                    )
                    
                    compute_forces_barnes_hut(
                        positions, masses, accelerations,
                        node_centers, node_half_sizes, node_masses, node_com,
                        node_children, node_body_idx, node_is_leaf,
                        num_nodes, num_bodies, theta, G, softening
                    )
                    
                    update_positions_velocities(
                        positions, velocities, accelerations,
                        damping, dt, num_bodies
                    )
                
                compute_colors_by_velocity(velocities, colors, num_bodies, 15.0)
            save_frame(rec_dir, frame, positions, colors)
            
            # Queue batch for background compression
            compressor.check_and_queue(frame)
            
            # Save state less frequently (every 50 frames) to reduce I/O
            if (frame + 1) % 50 == 0:
                np.savez(  # Uncompressed for speed
                    rec_dir / f"state_{frame:04d}.npz",
                    positions=positions,
                    velocities=velocities if use_gpu else velocities,
                )
                # Clean up old state files
                old_state = rec_dir / f"state_{frame-50:04d}.npz"
                if old_state.exists():
                    old_state.unlink()
            
            frame_time = time.time() - frame_start
            frame_times.append(frame_time)
            elapsed = time.time() - start_time
            
            avg_time = sum(frame_times[-10:]) / len(frame_times[-10:])
            remaining_frames = total_frames - frame - 1
            eta = avg_time * remaining_frames
            
            print_progress(frame, total_frames, frame_time, elapsed, eta)
        
        # Recording complete - compress remaining frames
        print(f"\n[Record] ✓ Recording complete!")
        print(f"[Record] Simulation time: {format_time(time.time() - start_time)}")
        
        # Compress any remaining uncompressed frames
        compressor.compress_remaining(total_frames)
        compressor.stop()
        
        print(f"[Record] Total time: {format_time(time.time() - start_time)}")
        print(f"[Record] Output: {rec_dir}")
        print(f"\n[Record] To playback: python -m tools.playback {config['session_name']}")
        
    except KeyboardInterrupt:
        print(f"\n[Record] Paused at frame {frame}")
        print(f"[Record] To resume: python -m tools.record --resume {config['session_name']}")
        
        # Stop background compressor and compress what we have
        print("[Record] Finishing compression...")
        compressor.compress_remaining(frame + 1)
        compressor.stop()
        
        np.savez_compressed(
            rec_dir / f"state_{frame:04d}.npz",
            positions=positions,
            velocities=velocities,
        )


def show_status(session_name: str):
    """Show recording status for a specific session."""
    rec_dir = get_recording_dir(session_name)
    
    if not (rec_dir / "metadata.json").exists():
        print(f"[Status] No recording found: {session_name}")
        return
    
    metadata = load_metadata(rec_dir)
    completed = get_completed_frames(rec_dir)
    total = metadata["total_frames"]
    pct = completed / total * 100
    
    print(f"\n[Status] Recording: {session_name}")
    print(f"  Bodies: {metadata['num_bodies']:,}")
    print(f"  Theta: {metadata['theta']}")
    print(f"  Distribution: {metadata.get('distribution', 'unknown')}")
    print(f"  Progress: {completed}/{total} frames ({pct:.1f}%)")
    print(f"  Started: {metadata.get('start_datetime', 'unknown')}")
    
    if completed < total:
        print(f"\n  To resume: python -m tools.record --resume {session_name}")
    else:
        print(f"\n  ✓ Complete! Playback: python -m tools.playback {session_name}")


def list_recordings():
    """List all available recordings."""
    recordings_dir = PROJECT_ROOT / "recordings"
    
    if not recordings_dir.exists():
        print("[List] No recordings directory found")
        return
    
    sessions = [d.name for d in recordings_dir.iterdir() if d.is_dir() and (d / "metadata.json").exists()]
    
    if not sessions:
        print("[List] No recordings found")
        return
    
    print(f"\n[List] Found {len(sessions)} recording(s):\n")
    
    for session in sorted(sessions):
        rec_dir = recordings_dir / session
        metadata = load_metadata(rec_dir)
        completed = get_completed_frames(rec_dir)
        total = metadata["total_frames"]
        pct = completed / total * 100
        status = "✓" if completed >= total else f"{pct:.0f}%"
        
        print(f"  {session:30s} | {metadata['num_bodies']:>10,} bodies | {completed:>4}/{total:<4} frames | {status}")
    
    print()


def main():
    parser = argparse.ArgumentParser(description="N-Body offline renderer")
    parser.add_argument("session", nargs="?", help="Session name (for --resume or --status)")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted recording")
    parser.add_argument("--status", action="store_true", help="Show recording status")
    parser.add_argument("--list", action="store_true", help="List all recordings")
    args = parser.parse_args()
    
    if args.list:
        list_recordings()
        return
    
    if args.session:
        session_name = args.session
    else:
        session_name = RECORDING_CONFIG["session_name"]
    
    if args.status:
        show_status(session_name)
    elif args.resume:
        config = RECORDING_CONFIG.copy()
        
        rec_dir = get_recording_dir(session_name)
        if (rec_dir / "metadata.json").exists():
            metadata = load_metadata(rec_dir)
            config.update(metadata)
        config["session_name"] = session_name
        
        record(config, resume=True)
    else:
        record(RECORDING_CONFIG, resume=False)


if __name__ == "__main__":
    main()

