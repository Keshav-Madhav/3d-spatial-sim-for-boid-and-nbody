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

# Import preset library
from tools.presets import (
    PRESETS, get_preset_list, print_preset_menu, 
    get_preset_by_index, get_preset_config, generate_distribution
)


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


def format_time(seconds: float, short: bool = False) -> str:
    """Format seconds as human-readable time.
    
    Args:
        seconds: Time in seconds
        short: If True, format for frame time (show ms for <1s)
               If False, format for elapsed time (stay in seconds until 90s)
    """
    if short:
        # Frame time format - show milliseconds for sub-second
        if seconds < 1.0:
            return f"{seconds*1000:.0f}ms"
        elif seconds < 90:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
    else:
        # Elapsed time format - stay in seconds until 90s
        if seconds < 90:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"


def format_eta(seconds: float) -> str:
    """Format ETA - stays in seconds until 90s, then switches to mm:ss or hh:mm:ss."""
    if seconds < 0:
        return "calculating..."
    if seconds < 90:
        return f"{seconds:.0f}s"
    # Use timedelta for longer durations
    td = timedelta(seconds=int(seconds))
    return str(td)


def print_progress(frame: int, total: int, frame_time: float, elapsed: float, eta: float):
    """Print progress for each frame on its own line."""
    pct = (frame + 1) / total * 100
    bar_width = 20
    filled = int(bar_width * (frame + 1) / total)
    bar = "█" * filled + "░" * (bar_width - filled)
    
    print(f"[{bar}] {pct:5.1f}% | Frame {frame+1:4d}/{total} | "
          f"Time: {format_time(frame_time, short=True):>6s} | Elapsed: {format_time(elapsed):>6s} | "
          f"ETA: {format_eta(eta)}")


def _generate_initial_conditions(config: dict):
    """Generate initial positions, velocities, masses based on distribution.
    
    Uses the distribution generator from presets library.
    This avoids creating a full NBodySimulation which would initialize GPU twice.
    """
    n = config["num_bodies"]
    R = config["spawn_radius"]
    G = config["G"]
    distribution = config.get("distribution", "galaxy")
    
    # Use preset library's distribution generator
    positions, velocities, masses = generate_distribution(distribution, n, R, G)
    
    # Convert to float64 for compatibility
    positions = positions.astype(np.float64)
    velocities = velocities.astype(np.float64)
    masses = masses.astype(np.float64)
    
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


def select_preset_interactive() -> dict:
    """Show preset menu and get user selection."""
    print_preset_menu()
    
    presets = get_preset_list()
    max_idx = len(presets) - 1
    
    while True:
        try:
            user_input = input("\n  Selection: ").strip().lower()
            
            if user_input in ['q', 'quit', 'exit']:
                print("\n  Cancelled.")
                return None
            
            idx = int(user_input)
            
            if 0 <= idx <= max_idx:
                key, preset = get_preset_by_index(idx)
                config = get_preset_config(key)
                
                print(f"\n  Selected: [{idx}] {preset['name']}")
                print(f"  Distribution: {config['distribution']}")
                print(f"  Bodies: {config['num_bodies']:,}")
                print(f"  Frames: {config['total_frames']}")
                print(f"  Estimated time: {preset.get('estimated_time', 'unknown')}")
                
                confirm = input("\n  Start recording? [Y/n]: ").strip().lower()
                if confirm in ['', 'y', 'yes']:
                    return config
                else:
                    print_preset_menu()
            else:
                print(f"  Invalid selection. Enter 0-{max_idx} or 'q' to quit.")
                
        except ValueError:
            print(f"  Invalid input. Enter a number 0-{max_idx} or 'q' to quit.")
        except KeyboardInterrupt:
            print("\n\n  Cancelled.")
            return None


def main():
    parser = argparse.ArgumentParser(description="N-Body offline renderer")
    parser.add_argument("session", nargs="?", help="Session name (for --resume or --status)")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted recording")
    parser.add_argument("--status", action="store_true", help="Show recording status")
    parser.add_argument("--list", action="store_true", help="List all recordings")
    parser.add_argument("--preset", type=str, help="Use preset by name (e.g., 'quick_galaxy')")
    parser.add_argument("--preset-id", type=int, help="Use preset by index number")
    args = parser.parse_args()
    
    if args.list:
        list_recordings()
        return
    
    # Handle status check
    if args.status:
        if args.session:
            show_status(args.session)
        else:
            list_recordings()
        return
    
    # Handle resume
    if args.resume:
        if args.session:
            session_name = args.session
        else:
            # Find most recent recording to resume
            recordings_dir = PROJECT_ROOT / "recordings"
            if recordings_dir.exists():
                sessions = [d for d in recordings_dir.iterdir() 
                           if d.is_dir() and (d / "metadata.json").exists()]
                if sessions:
                    # Sort by modification time
                    sessions.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    session_name = sessions[0].name
                    print(f"[Record] Resuming most recent: {session_name}")
                else:
                    print("[Record] No recordings found to resume")
                    return
            else:
                print("[Record] No recordings directory found")
                return
        
        rec_dir = get_recording_dir(session_name)
        if not (rec_dir / "metadata.json").exists():
            print(f"[Record] No metadata found for session: {session_name}")
            return
        
        config = load_metadata(rec_dir)
        config["session_name"] = session_name
        record(config, resume=True)
        return
    
    # Handle preset selection
    config = None
    
    if args.preset_id is not None:
        # Use preset by index
        key, preset = get_preset_by_index(args.preset_id)
        if key:
            config = get_preset_config(key)
            print(f"[Record] Using preset [{args.preset_id}]: {preset['name']}")
        else:
            print(f"[Record] Invalid preset index: {args.preset_id}")
            return
    elif args.preset:
        # Use preset by name
        config = get_preset_config(args.preset)
        if config:
            print(f"[Record] Using preset: {args.preset}")
        else:
            print(f"[Record] Unknown preset: {args.preset}")
            print("[Record] Available presets:")
            for key in sorted(PRESETS.keys()):
                print(f"  - {key}")
            return
    else:
        # Interactive menu
        config = select_preset_interactive()
        if config is None:
            return
    
    record(config, resume=False)


if __name__ == "__main__":
    main()

