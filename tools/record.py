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
        frame_0000.zstd   - Compressed position/color data (zstd+delta compression)
        ...
"""

import os
import sys
import json
import time
import shutil
import argparse
import struct
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
    while True:
        # Check for uncompressed .npz (during recording) or compressed .zstd (after compression)
        if (rec_dir / f"frame_{count:04d}.npz").exists() or (rec_dir / f"frame_{count:04d}.zstd").exists():
            count += 1
        else:
            break
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


def load_frame(rec_dir: Path, frame_idx: int, prev_positions: np.ndarray = None,
               prev_colors: np.ndarray = None) -> tuple:
    """
    Load a single frame from disk.
    
    If the frame uses delta compression and prev_positions/prev_colors are not provided,
    this function will automatically load the previous frame(s) as needed.
    
    Uses iterative loading instead of recursion to avoid stack overflow.
    
    Args:
        rec_dir: Recording directory
        frame_idx: Frame index
        prev_positions: Previous frame positions (for delta decompression, auto-loaded if None)
        prev_colors: Previous frame colors (for delta decompression, auto-loaded if None)
    
    Returns:
        (positions, colors) tuple
    """
    zstd_file = rec_dir / f"frame_{frame_idx:04d}.zstd"
    npz_file = rec_dir / f"frame_{frame_idx:04d}.npz"
    
    if zstd_file.exists():
        # Compressed zstd format
        with open(zstd_file, 'rb') as f:
            compressed_data = f.read()
        
        if len(compressed_data) > 0:
            comp_format = struct.unpack('B', compressed_data[0:1])[0]
            
            # If delta-compressed and we don't have previous frame, load it iteratively
            if comp_format == 2 and (prev_positions is None or prev_colors is None):
                if frame_idx > 0:
                    # Iteratively load frames backwards until we find a base frame (format 1)
                    # This avoids recursion depth issues
                    current_idx = frame_idx - 1
                    prev_positions = None
                    prev_colors = None
                    
                    # Collect frames we need to decompress (in reverse order)
                    frames_to_decompress = []
                    
                    # Load frames backwards until we find a base frame (format 1) or uncompressed frame
                    while current_idx >= 0:
                        prev_zstd = rec_dir / f"frame_{current_idx:04d}.zstd"
                        prev_npz = rec_dir / f"frame_{current_idx:04d}.npz"
                        
                        if prev_zstd.exists():
                            with open(prev_zstd, 'rb') as f:
                                prev_data = f.read()
                            if len(prev_data) > 0:
                                prev_format = struct.unpack('B', prev_data[0:1])[0]
                                frames_to_decompress.append((current_idx, prev_data, prev_format))
                                
                                # If this is a base frame (format 1), we can stop
                                if prev_format == 1:
                                    break
                                
                                # Otherwise, continue backwards
                                current_idx -= 1
                            else:
                                break
                        elif prev_npz.exists():
                            # Uncompressed frame - load it directly as base
                            with np.load(prev_npz) as data:
                                prev_positions = data["positions"].copy()
                                prev_colors = data["colors"].copy()
                            break
                        else:
                            raise FileNotFoundError(f"Frame {current_idx:04d} not found (needed for delta decompression)")
                    
                    # If we collected frames to decompress, decompress them forward from base
                    if prev_positions is None and frames_to_decompress:
                        # frames_to_decompress is collected backwards: [frame_idx-1, frame_idx-2, ..., 0]
                        # So frames_to_decompress[0] = frame_idx-1, frames_to_decompress[-1] = 0
                        # We need to decompress forward: 0 -> 1 -> ... -> frame_idx-1
                        # So iterate from last to first (reverse order)
                        
                        base_found = False
                        base_idx = -1
                        
                        # Find the base frame (format 1) - should be frame 0 or earliest format 1 frame
                        for i in range(len(frames_to_decompress) - 1, -1, -1):
                            idx, data, fmt = frames_to_decompress[i]
                            if fmt == 1:
                                # This is the base frame - decompress it first
                                prev_positions, prev_colors = decompress_frame(data, None, None)
                                base_found = True
                                base_idx = idx
                                break
                        
                        if not base_found:
                            raise ValueError(f"Frame {frame_idx:04d} appears to be delta-compressed but no base frame (format 1) found")
                        
                        # Decompress frames forward from the base up to frame_idx-1
                        # Iterate from base_idx+1 to frame_idx-1 (forward order)
                        for i in range(len(frames_to_decompress) - 1, -1, -1):
                            idx, data, fmt = frames_to_decompress[i]
                            if idx > base_idx:  # Decompress frames after the base
                                prev_positions, prev_colors = decompress_frame(data, prev_positions, prev_colors)
                    elif prev_positions is None:
                        raise ValueError(f"Frame {frame_idx:04d} appears to be delta-compressed but no base frame found")
                else:
                    raise ValueError(f"Frame {frame_idx:04d} appears to be delta-compressed but is the first frame")
        
        return decompress_frame(compressed_data, prev_positions, prev_colors)
    elif npz_file.exists():
        # Uncompressed format (during recording, before compression)
        with np.load(npz_file) as data:
            return data["positions"].copy(), data["colors"].copy()
    else:
        raise FileNotFoundError(f"Frame {frame_idx:04d} not found")


# =============================================================================
# BACKGROUND BATCH COMPRESSION WITH ZSTD + DELTA COMPRESSION
# =============================================================================

import threading
from queue import Queue
import shutil
import struct
import gc

# Compression batch size - compress this many frames at once
# Smaller batches = less memory pressure, more frequent cleanup
COMPRESSION_BATCH_SIZE = 50

# zstandard is required
import zstandard as zstd


def compress_frame(positions: np.ndarray, colors: np.ndarray, 
                   prev_positions: np.ndarray = None, 
                   prev_colors: np.ndarray = None) -> bytes:
    """
    Compress frame data using zstd with delta compression.
    
    Delta compression stores differences between frames, which compresses
    much better than absolute values for smooth animations.
    
    Format:
    - 1 byte: compression format (1=zstd absolute, 2=zstd+delta)
    - 4 bytes: positions data size
    - N bytes: compressed positions
    - 4 bytes: colors data size  
    - N bytes: compressed colors
    """
    # Use delta compression if previous frame is available, otherwise absolute
    use_delta = prev_positions is not None and prev_colors is not None
    comp_format = 2 if use_delta else 1
    
    # Use level 19 (max compression) for best compression ratio
    cctx = zstd.ZstdCompressor(level=19, threads=1)
    
    if use_delta:
        # Delta compression: store differences instead of absolute values
        pos_delta = positions - prev_positions
        col_delta = colors - prev_colors
        # Convert to int16 for better compression (differences are small)
        pos_delta_int = (pos_delta * 1000).astype(np.int16)  # Scale to preserve precision
        col_delta_int = (col_delta * 1000).astype(np.int16)
        pos_data = pos_delta_int.tobytes()
        col_data = col_delta_int.tobytes()
    else:
        # Absolute values (first frame only)
        pos_data = positions.astype(np.float32).tobytes()
        col_data = colors.astype(np.float32).tobytes()
    
    # Compress with zstd
    pos_compressed = cctx.compress(pos_data)
    col_compressed = cctx.compress(col_data)
    
    # Pack format: format byte, positions size (4 bytes), positions data, colors size (4 bytes), colors data
    result = struct.pack('B', comp_format)
    result += struct.pack('I', len(pos_compressed))
    result += pos_compressed
    result += struct.pack('I', len(col_compressed))
    result += col_compressed
    
    return result


def decompress_frame(data: bytes, prev_positions: np.ndarray = None,
                     prev_colors: np.ndarray = None) -> tuple:
    """
    Decompress frame data using zstd with delta compression.
    """
    if len(data) < 1:
        raise ValueError("Invalid compressed data")
    
    comp_format = struct.unpack('B', data[0:1])[0]
    offset = 1
    
    # Read positions
    pos_size = struct.unpack('I', data[offset:offset+4])[0]
    offset += 4
    pos_compressed = data[offset:offset+pos_size]
    offset += pos_size
    
    # Read colors
    col_size = struct.unpack('I', data[offset:offset+4])[0]
    offset += 4
    col_compressed = data[offset:offset+col_size]
    
    # Decompress with zstd
    dctx = zstd.ZstdDecompressor()
    pos_data = dctx.decompress(pos_compressed)
    col_data = dctx.decompress(col_compressed)
    
    if comp_format == 1:
        # Absolute values (first frame)
        positions = np.frombuffer(pos_data, dtype=np.float32).reshape(-1, 3)
        colors = np.frombuffer(col_data, dtype=np.float32).reshape(-1, 3)
    elif comp_format == 2:
        # Delta compression - reconstruct from differences
        if prev_positions is None or prev_colors is None:
            raise ValueError("Delta compression requires previous frame")
        pos_delta_int = np.frombuffer(pos_data, dtype=np.int16).reshape(-1, 3)
        col_delta_int = np.frombuffer(col_data, dtype=np.int16).reshape(-1, 3)
        pos_delta = pos_delta_int.astype(np.float32) / 1000.0
        col_delta = col_delta_int.astype(np.float32) / 1000.0
        positions = prev_positions + pos_delta
        colors = prev_colors + col_delta
    else:
        raise ValueError(f"Unknown compression format: {comp_format}")
    
    return positions, colors


class BackgroundCompressor:
    """
    Compresses frames in batches in the background while recording continues.
    
    Uses zstd compression with delta encoding for optimal compression ratios.
    
    Strategy:
    - Frames are saved uncompressed for speed (~4ms each)
    - Every BATCH_SIZE frames, queue them for background compression
    - Background thread compresses with zstd+delta and replaces uncompressed files
    - Delta compression stores differences between frames (much smaller)
    """
    
    def __init__(self, rec_dir: Path, batch_size: int = COMPRESSION_BATCH_SIZE):
        self.rec_dir = rec_dir
        self.batch_size = batch_size
        self.queue = Queue()
        self.thread = None
        self.running = False
        self.compressed_count = 0
        self.pending_batches = 0
        self.total_saved_bytes = 0
        self.total_original_bytes = 0
        self.total_frames_to_compress = 0
        self.lock = threading.Lock()  # For thread-safe progress updates
    
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
            self.thread.join(timeout=300)  # Wait up to 5min for completion (compression can be slow)
    
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
                
                with self.lock:
                    self.pending_batches -= 1
                
                # Force garbage collection after each batch to free memory
                gc.collect()
                
            except:
                continue
    
    def _compress_batch(self, start_frame: int, end_frame: int):
        """Compress a batch of frames with delta compression."""
        prev_positions = None
        prev_colors = None
        
        # Load previous frame if needed for delta compression
        if start_frame > 0:
            try:
                prev_positions, prev_colors = load_frame(self.rec_dir, start_frame - 1)
            except:
                # Previous frame not available, start without delta
                prev_positions = None
                prev_colors = None
        
        for frame_idx in range(start_frame, end_frame):
            uncompressed = self.rec_dir / f"frame_{frame_idx:04d}.npz"
            
            if not uncompressed.exists():
                # Frame might already be compressed, try to load it for next delta
                try:
                    prev_positions, prev_colors = load_frame(self.rec_dir, frame_idx)
                except:
                    pass
                continue
            
            try:
                # Load uncompressed - MUST use context manager to close file!
                with np.load(uncompressed) as data:
                    positions = data['positions'].copy()  # Copy before file closes
                    colors = data['colors'].copy()
                
                # Get original size
                original_size = uncompressed.stat().st_size
                self.total_original_bytes += original_size
                
                # Compress with zstd+delta
                compressed_data = compress_frame(
                    positions, colors, 
                    prev_positions,
                    prev_colors
                )
                
                # Save compressed data
                compressed_file = self.rec_dir / f"frame_{frame_idx:04d}.zstd"
                with open(compressed_file, 'wb') as f:
                    f.write(compressed_data)
                
                # Remove uncompressed file
                uncompressed.unlink()
                
                # Track compression stats
                compressed_size = len(compressed_data)
                self.total_saved_bytes += compressed_size
                self.compressed_count += 1
                
                # Store for delta compression of next frame
                prev_positions = positions.copy()
                prev_colors = colors.copy()
                
                # Explicitly free memory
                del positions, colors
                
            except Exception as e:
                # If compression fails, keep the uncompressed file
                import traceback
                print(f"[Compress] Warning: Failed to compress frame {frame_idx}: {e}")
                traceback.print_exc()
    
    def get_compressed_count(self, total_frames: int) -> int:
        """Get current count of compressed frames (thread-safe)."""
        compressed_count = 0
        for i in range(total_frames):
            if (self.rec_dir / f"frame_{i:04d}.zstd").exists():
                compressed_count += 1
        return compressed_count
    
    def compress_remaining(self, total_frames: int, start_time: float, frame_times: list):
        """
        Compress any remaining uncompressed frames after recording.
        Uses the same nested progress bar as frame generation.
        """
        # Find the last queued batch end
        last_batch_end = (total_frames // self.batch_size) * self.batch_size
        
        # Compress remaining frames (if any)
        if last_batch_end < total_frames:
            self._compress_batch(last_batch_end, total_frames)
        
        # Wait for background thread to finish with progress updates
        # Continue using the same nested progress bar
        import time
        last_update = 0
        
        # All frames are generated, so frame = total_frames - 1
        final_frame = total_frames - 1
        
        # Wait while there are pending batches or uncompressed frames
        while True:
            with self.lock:
                pending = self.pending_batches
            
            # Count current progress
            compressed_now = self.get_compressed_count(total_frames)
            
            # Break if all frames are compressed and no batches pending
            if compressed_now == total_frames and pending == 0:
                # Give thread a moment to finish, then break
                time.sleep(0.2)
                with self.lock:
                    if self.pending_batches == 0:
                        break
            
            # Update progress every 0.5s to avoid overhead
            if time.time() - last_update >= 0.5:
                elapsed = time.time() - start_time
                # Use average frame time for display (compression doesn't have frame_time)
                avg_frame_time = sum(frame_times[-10:]) / len(frame_times[-10:]) if frame_times else 0.0
                eta = 0.0  # No ETA during compression
                
                # Continue using the same nested progress bar
                # Frame generation is complete, compression is progressing
                print_progress(final_frame, total_frames, avg_frame_time, elapsed, eta,
                              compressed=compressed_now, compressor=self)
                
                last_update = time.time()
            
            time.sleep(0.1)
    
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


def print_progress(frame: int, total: int, frame_time: float, elapsed: float, eta: float,
                   compressed: int = None, compressor: 'BackgroundCompressor' = None):
    """
    Print nested progress bars (frame generation + compression overlay) and details.
    """
    pct = (frame + 1) / total * 100
    
    # Get terminal width, default to 80 if can't detect
    try:
        term_width = shutil.get_terminal_size().columns
    except:
        term_width = 80
    
    # Progress bar on first line (full width minus brackets)
    bar_width = term_width - 2  # Account for [ and ]
    frame_filled = int(bar_width * (frame + 1) / total)
    
    # Build nested progress bar
    # Base bar: frame generation progress
    bar_chars = ["░"] * bar_width
    
    # Overlay compression progress on top (if compression is happening)
    if compressed is not None and compressed >= 0:
        compressed_filled = int(bar_width * compressed / total) if total > 0 else 0
        # Compression can't exceed generated frames
        compressed_filled = min(compressed_filled, frame_filled)
        
        for i in range(compressed_filled):
            bar_chars[i] = "█"  # Compressed frames (overlay)
        for i in range(compressed_filled, frame_filled):
            bar_chars[i] = "▒"  # Generated but not compressed
    else:
        # No compression info, just show frame generation
        for i in range(frame_filled):
            bar_chars[i] = "█"
    
    bar = "".join(bar_chars)
    
    # Details on second line
    compression_info = ""
    if compressed is not None and compressed >= 0:
        compression_info = f" | Compressed: {compressed}/{total}"
    
    details = (f"{pct:5.1f}% | Frame {frame+1:4d}/{total}{compression_info} | "
               f"Time: {format_time(frame_time, short=True):>6s} | "
               f"Elapsed: {format_time(elapsed):>6s} | ETA: {format_eta(eta)}")
    
    # Move cursor up 2 lines, clear, and print both lines
    # Use ANSI escape codes: \033[2A moves up 2 lines, \033[K clears line
    if frame > 0:
        sys.stdout.write("\033[2A")  # Move up 2 lines
    
    sys.stdout.write(f"\033[K[{bar}]\n")  # Clear line and print bar
    sys.stdout.write(f"\033[K{details}\n")  # Clear line and print details
    sys.stdout.flush()


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
            
            # Get compression progress for nested progress bar
            compressed_count = compressor.get_compressed_count(total_frames)
            print_progress(frame, total_frames, frame_time, elapsed, eta, 
                          compressed=compressed_count, compressor=compressor)
        
        # Frame generation complete - compress remaining frames
        # Continue showing nested progress bar during compression
        compressor.compress_remaining(total_frames, start_time, frame_times)
        compressor.stop()
        
        # Clear the progress bar and show completion
        sys.stdout.write("\033[2A")  # Move up 2 lines (progress bar + details)
        sys.stdout.write("\033[K")  # Clear progress bar line
        sys.stdout.write("\033[K")  # Clear details line
        sys.stdout.flush()
        
        # Print compression stats
        compression_ratio = (1 - compressor.total_saved_bytes / compressor.total_original_bytes) * 100 if compressor.total_original_bytes > 0 else 0
        avg_size_mb = compressor.total_saved_bytes / max(1, compressor.compressed_count) / (1024 * 1024)
        
        print(f"[Record] ✓ Recording complete!")
        print(f"[Record] Simulation time: {format_time(time.time() - start_time)}")
        print(f"[Compress] Compressed {compressor.compressed_count}/{total_frames} frames")
        print(f"[Compress] Compression ratio: {compression_ratio:.1f}% reduction")
        print(f"[Compress] Average frame size: {avg_size_mb:.2f} MB")
        print(f"[Record] Total time: {format_time(time.time() - start_time)}")
        print(f"[Record] Output: {rec_dir}")
        print(f"\n[Record] To playback: python -m tools.playback {config['session_name']}")
        
    except KeyboardInterrupt:
        # Clear progress bar
        sys.stdout.write("\033[2A")  # Move up 2 lines
        sys.stdout.write("\033[K")  # Clear progress bar line
        sys.stdout.write("\033[K")  # Clear details line
        sys.stdout.flush()
        
        print(f"\n[Record] Paused at frame {frame}")
        print(f"[Record] To resume: python -m tools.record --resume {config['session_name']}")
        
        # Stop background compressor and compress what we have
        print("[Record] Finishing compression...")
        compressor.compress_remaining(frame + 1, start_time, frame_times)
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


def estimate_render_time(bodies: int, theta: float, frames: int, substeps: int) -> str:
    """Estimate render time based on parameters."""
    import math
    base_bodies = 100_000
    base_theta = 0.8
    base_time_ms = 70  # ms per physics step from benchmark
    
    body_factor = (bodies * math.log(bodies)) / (base_bodies * math.log(base_bodies))
    theta_factor = (base_theta / min(theta, 1.5)) ** 2
    time_per_step_ms = base_time_ms * body_factor * theta_factor
    total_steps = frames * substeps
    seconds = time_per_step_ms * total_steps / 1000
    
    if seconds < 45:
        return f"~{int(seconds)} seconds"
    elif seconds < 90:
        return "~1 minute"
    elif seconds < 150:
        return "~2 minutes"
    elif seconds < 3600:
        return f"~{seconds / 60:.0f} minutes"
    elif seconds < 7200:
        return "~1 hour"
    else:
        return f"~{seconds / 3600:.0f} hours"


def select_preset_interactive() -> dict:
    """Show preset menu and get user selection with optional overrides."""
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
                print(f"  Theta: {config['theta']}")
                print(f"  Estimated time: {preset.get('estimated_time', 'unknown')}")
                
                # Optional overrides
                print(f"\n  ─── Optional Overrides (press Enter to skip) ───")
                
                # Bodies override
                bodies_input = input(f"  Bodies [{config['num_bodies']:,}]: ").strip()
                if bodies_input:
                    try:
                        new_bodies = parse_number(bodies_input)
                        if new_bodies > 0:
                            config['num_bodies'] = new_bodies
                            print(f"    → Bodies set to {new_bodies:,}")
                    except ValueError:
                        print(f"    → Invalid, keeping {config['num_bodies']:,}")
                
                # Frames override
                frames_input = input(f"  Frames [{config['total_frames']}]: ").strip()
                if frames_input:
                    try:
                        new_frames = int(frames_input)
                        if new_frames > 0:
                            config['total_frames'] = new_frames
                            print(f"    → Frames set to {new_frames}")
                    except ValueError:
                        print(f"    → Invalid, keeping {config['total_frames']}")
                
                # Theta override
                theta_input = input(f"  Theta [{config['theta']}]: ").strip()
                if theta_input:
                    try:
                        new_theta = float(theta_input)
                        if 0.1 <= new_theta <= 2.0:
                            config['theta'] = new_theta
                            print(f"    → Theta set to {new_theta}")
                        else:
                            print(f"    → Theta must be 0.1-2.0, keeping {config['theta']}")
                    except ValueError:
                        print(f"    → Invalid, keeping {config['theta']}")
                
                # Recalculate time estimate
                new_estimate = estimate_render_time(
                    config['num_bodies'], 
                    config['theta'], 
                    config['total_frames'], 
                    config.get('substeps', 1)
                )
                
                # Show final config
                print(f"\n  ─── Final Configuration ───")
                print(f"  Bodies: {config['num_bodies']:,}")
                print(f"  Frames: {config['total_frames']}")
                print(f"  Theta: {config['theta']}")
                print(f"  Estimated time: {new_estimate}")
                
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


def parse_number(value: str) -> int:
    """Parse number with optional suffix (k, m, K, M)."""
    value = value.strip().lower()
    multipliers = {'k': 1_000, 'm': 1_000_000}
    
    for suffix, mult in multipliers.items():
        if value.endswith(suffix):
            return int(float(value[:-1]) * mult)
    
    return int(value)


def main():
    parser = argparse.ArgumentParser(description="N-Body offline renderer")
    parser.add_argument("session", nargs="?", help="Session name (for --resume, --status, or --extend)")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted recording")
    parser.add_argument("--extend", type=int, metavar="FRAMES", help="Extend existing recording by N frames (e.g., --extend 3000)")
    parser.add_argument("--status", action="store_true", help="Show recording status")
    parser.add_argument("--list", action="store_true", help="List all recordings")
    parser.add_argument("--preset", type=str, help="Use preset by name (e.g., 'quick_galaxy')")
    parser.add_argument("--preset-id", type=int, help="Use preset by index number")
    parser.add_argument("--bodies", "-n", type=str, help="Override number of bodies (e.g., 100000, 100k, 1m)")
    parser.add_argument("--frames", "-f", type=int, help="Override number of frames")
    parser.add_argument("--theta", "-t", type=float, help="Override Barnes-Hut theta (0.3-1.5)")
    parser.add_argument("--dt", type=float, help="Override time step")
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
    
    # Handle extend (add more frames to existing recording)
    if args.extend:
        if not args.session:
            print("[Record] Error: --extend requires a session name")
            print("[Record] Usage: python -m tools.record --extend 3000 bar_galaxy")
            return
        
        session_name = args.session
        rec_dir = get_recording_dir(session_name)
        
        if not (rec_dir / "metadata.json").exists():
            print(f"[Record] No recording found: {session_name}")
            return
        
        # Load and update metadata
        config = load_metadata(rec_dir)
        old_frames = config["total_frames"]
        new_frames = old_frames + args.extend
        
        completed = get_completed_frames(rec_dir)
        
        print(f"[Record] Extending '{session_name}'")
        print(f"  Current frames: {completed}/{old_frames}")
        print(f"  Adding: +{args.extend} frames")
        print(f"  New total: {new_frames} frames")
        
        # Calculate new time estimate
        new_estimate = estimate_render_time(
            config['num_bodies'],
            config['theta'],
            args.extend,  # Only estimate time for new frames
            config.get('substeps', 1)
        )
        print(f"  Estimated time for new frames: {new_estimate}")
        
        # Update and save metadata
        config["total_frames"] = new_frames
        with open(rec_dir / "metadata.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"\n[Record] ✓ Updated metadata. Now resuming...")
        
        config["session_name"] = session_name
        record(config, resume=True)
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
    
    # Apply overrides from command line
    if args.bodies:
        try:
            config["num_bodies"] = parse_number(args.bodies)
            print(f"[Record] Override: {config['num_bodies']:,} bodies")
        except ValueError:
            print(f"[Record] Invalid bodies value: {args.bodies}")
            return
    
    if args.frames:
        config["total_frames"] = args.frames
        print(f"[Record] Override: {args.frames} frames")
    
    if args.theta:
        config["theta"] = args.theta
        print(f"[Record] Override: θ={args.theta}")
    
    if args.dt:
        config["dt"] = args.dt
        print(f"[Record] Override: dt={args.dt}")
    
    record(config, resume=False)


if __name__ == "__main__":
    main()

