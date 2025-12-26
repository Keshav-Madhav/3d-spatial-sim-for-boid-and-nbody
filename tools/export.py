"""
N-Body Recording Video Exporter
===============================

Export recorded N-body simulations to high-quality, optimized MP4 videos.
Uses FFmpeg for superior compression without quality loss.

Usage:
    python -m tools.export <session_name>                    # Export with defaults
    python -m tools.export <session_name> --fps 60           # Custom FPS
    python -m tools.export <session_name> --quality high     # Quality preset
    python -m tools.export <session_name> --resolution 4k    # 4K resolution
    python -m tools.export <session_name> --camera orbit     # Camera animation

Quality Presets:
    fast     - Quick export, larger file (~10MB/min)
    balanced - Good balance of speed and size (~5MB/min)
    high     - Best quality, smaller file, slower export (~3MB/min)
    lossless - Visually lossless, moderate size (~8MB/min)

Camera Modes:
    fixed    - Static camera position
    orbit    - Slow horizontal orbit
    spiral   - Spiral around the simulation
    zoom     - Slow zoom in/out cycle
    zoomout  - Constant zoom out
    zoomin   - Constant zoom in
    cinematic - Dramatic slow sweep
"""

import os
import sys
import json
import argparse
import subprocess
import shutil
import tempfile
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

# Get project root (parent of tools/)
PROJECT_ROOT = Path(__file__).parent.parent

# Import shared functions from record module
from tools.record import get_recording_dir, load_metadata, get_completed_frames as get_frame_count, load_frame


# =============================================================================
# EXPORT CONFIGURATION
# =============================================================================

@dataclass
class ExportConfig:
    """Export configuration options."""
    # Video settings
    fps: int = 30
    resolution: Tuple[int, int] = (1920, 1080)
    
    # Quality settings (CRF: 0=lossless, 18=high, 23=balanced, 28=low)
    quality_preset: str = "balanced"
    crf: int = 23
    encoding_preset: str = "medium"  # ultrafast, fast, medium, slow, veryslow
    
    # Camera settings
    camera_mode: str = "orbit"
    camera_rotation_speed: float = 0.3  # degrees per frame
    camera_initial_theta: float = 45.0
    camera_initial_phi: float = 25.0
    camera_radius: float = 800.0
    
    # Rendering settings
    point_size: float = 1.5
    background_color: Tuple[float, float, float] = (0.0, 0.0, 0.02)
    fog_density: float = 0.0003
    
    # Frame range
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    
    # Output
    output_path: Optional[str] = None
    codec: str = "h264"  # h264, h265, vp9


# Quality presets
QUALITY_PRESETS = {
    "fast": {
        "crf": 28,
        "encoding_preset": "fast",
        "description": "Quick export, larger file size"
    },
    "balanced": {
        "crf": 23,
        "encoding_preset": "medium",
        "description": "Good balance of speed and quality"
    },
    "high": {
        "crf": 18,
        "encoding_preset": "slow",
        "description": "High quality, smaller file, slower export"
    },
    "lossless": {
        "crf": 15,
        "encoding_preset": "slow",  # "veryslow" is too slow for negligible gain
        "description": "Visually lossless, best compression"
    }
}

# Resolution presets
RESOLUTION_PRESETS = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "4k": (3840, 2160),
    "ultrawide": (2560, 1080),
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
# (get_recording_dir, load_metadata, get_frame_count, load_frame imported from tools.record)


def check_ffmpeg() -> bool:
    """Check if FFmpeg is available."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def format_time(seconds: float) -> str:
    """Format seconds as human-readable time."""
    if seconds < 90:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def format_size(bytes_size: int) -> str:
    """Format file size."""
    if bytes_size < 1024:
        return f"{bytes_size}B"
    elif bytes_size < 1024 * 1024:
        return f"{bytes_size / 1024:.1f}KB"
    elif bytes_size < 1024 * 1024 * 1024:
        return f"{bytes_size / (1024 * 1024):.1f}MB"
    else:
        return f"{bytes_size / (1024 * 1024 * 1024):.2f}GB"


# =============================================================================
# CAMERA
# =============================================================================

class ExportCamera:
    """Camera for video export with various animation modes."""
    
    def __init__(self, config: ExportConfig):
        self.config = config
        self.radius = config.camera_radius
        self.theta = config.camera_initial_theta
        self.phi = config.camera_initial_phi
        self.target = np.array([0.0, 0.0, 0.0])
        self.frame = 0
        
    def update(self, frame_idx: int, total_frames: int):
        """Update camera position based on animation mode."""
        self.frame = frame_idx
        t = frame_idx / max(1, total_frames - 1)  # 0 to 1
        
        mode = self.config.camera_mode
        speed = self.config.camera_rotation_speed
        
        if mode == "fixed":
            # No animation
            pass
        elif mode == "orbit":
            # Horizontal orbit
            self.theta = self.config.camera_initial_theta + frame_idx * speed
        elif mode == "spiral":
            # Spiral around with slight vertical change
            self.theta = self.config.camera_initial_theta + frame_idx * speed
            self.phi = self.config.camera_initial_phi + 10 * np.sin(t * 2 * np.pi)
        elif mode == "zoom":
            # Zoom in/out cycle
            self.theta = self.config.camera_initial_theta + frame_idx * speed * 0.5
            zoom_factor = 1.0 + 0.3 * np.sin(t * 2 * np.pi)
            self.radius = self.config.camera_radius * zoom_factor
        elif mode == "zoomout":
            # Constant zoom out (starts close, ends far)
            self.theta = self.config.camera_initial_theta + frame_idx * speed * 0.2
            # Zoom from 0.5x to 2.5x of initial radius
            zoom_factor = 0.5 + 2.0 * t
            self.radius = self.config.camera_radius * zoom_factor
        elif mode == "zoomin":
            # Constant zoom in (starts far, ends close)
            self.theta = self.config.camera_initial_theta + frame_idx * speed * 0.4
            # Zoom from 2.0x to 0.4x of initial radius
            zoom_factor = 2.0 - 2.0 * t
            self.radius = self.config.camera_radius * zoom_factor
        elif mode == "cinematic":
            # Dramatic slow sweep
            self.theta = self.config.camera_initial_theta + frame_idx * speed * 0.3
            self.phi = self.config.camera_initial_phi + 15 * np.sin(t * np.pi)
            self.radius = self.config.camera_radius * (1.0 - 0.2 * t)
        elif mode == "flyby":
            # Dramatic flyby - camera moves past the simulation
            self.theta = self.config.camera_initial_theta + 90 * t
            self.phi = self.config.camera_initial_phi - 20 + 40 * t
            self.radius = self.config.camera_radius * (1.5 - 0.8 * np.sin(t * np.pi))
        elif mode == "topdown":
            # Top-down view with slow rotation
            self.theta = self.config.camera_initial_theta + frame_idx * speed * 0.5
            self.phi = 80  # Nearly top-down
            self.radius = self.config.camera_radius * 1.2
    
    def get_position(self) -> np.ndarray:
        import math
        theta_rad = math.radians(self.theta)
        phi_rad = math.radians(self.phi)
        x = self.radius * math.cos(phi_rad) * math.cos(theta_rad)
        y = self.radius * math.sin(phi_rad)
        z = self.radius * math.cos(phi_rad) * math.sin(theta_rad)
        return np.array([x, y, z])
    
    def get_up_vector(self) -> tuple:
        import math
        phi_rad = math.radians(self.phi)
        if math.cos(phi_rad) >= 0:
            return (0, 1, 0)
        else:
            return (0, -1, 0)
    
    def apply(self):
        from OpenGL.GL import glLoadIdentity
        from OpenGL.GLU import gluLookAt
        
        pos = self.get_position()
        up = self.get_up_vector()
        glLoadIdentity()
        gluLookAt(
            pos[0], pos[1], pos[2],
            self.target[0], self.target[1], self.target[2],
            up[0], up[1], up[2]
        )


# =============================================================================
# VIDEO EXPORTER
# =============================================================================

class VideoExporter:
    """High-quality video exporter using OpenGL rendering and FFmpeg encoding."""
    
    def __init__(self, session_name: str, config: ExportConfig):
        self.session_name = session_name
        self.config = config
        
        self.rec_dir = get_recording_dir(session_name)
        if not self.rec_dir.exists():
            raise FileNotFoundError(f"Recording not found: {session_name}")
        
        self.metadata = load_metadata(self.rec_dir)
        self.total_frames = get_frame_count(self.rec_dir)
        
        if self.total_frames == 0:
            raise ValueError(f"No frames found in recording: {session_name}")
        
        # Determine frame range
        self.start_frame = config.start_frame or 0
        self.end_frame = config.end_frame or self.total_frames
        self.end_frame = min(self.end_frame, self.total_frames)
        self.export_frames = self.end_frame - self.start_frame
        
        # Output path - store in recordings/ folder directly (not inside session folder)
        if config.output_path:
            self.output_path = Path(config.output_path)
        else:
            self.output_path = self._get_unique_output_path(session_name)
        
        self.width, self.height = config.resolution
        self.camera = ExportCamera(config)
        
        # Temp directory for frame export
        self.temp_dir = None
        self.use_pipe = True  # Pipe frames directly to FFmpeg
    
    def _get_unique_output_path(self, session_name: str) -> Path:
        """Get unique output path, adding (1), (2), etc. if file exists."""
        recordings_dir = PROJECT_ROOT / "recordings"
        base_path = recordings_dir / f"{session_name}.mp4"
        
        if not base_path.exists():
            return base_path
        
        # File exists, find next available number
        counter = 1
        while True:
            new_path = recordings_dir / f"{session_name} ({counter}).mp4"
            if not new_path.exists():
                return new_path
            counter += 1
    
    def _setup_opengl(self):
        """Initialize PyGame and OpenGL."""
        import pygame
        from pygame.locals import DOUBLEBUF, OPENGL
        from OpenGL.GL import (
            glClearColor, glEnable, glFogi, glFogfv, glFogf,
            glMatrixMode, glLoadIdentity, GL_DEPTH_TEST, GL_FOG,
            GL_FOG_MODE, GL_EXP2, GL_FOG_COLOR, GL_FOG_DENSITY,
            GL_PROJECTION, GL_MODELVIEW
        )
        from OpenGL.GLU import gluPerspective
        
        pygame.init()
        
        # Create window for OpenGL context (required for rendering)
        # Position off-screen and minimize to reduce resource usage
        os.environ['SDL_VIDEO_WINDOW_POS'] = '-10000,-10000'  # Off-screen
        pygame.display.set_mode(
            (self.width, self.height),
            DOUBLEBUF | OPENGL | pygame.NOFRAME  # No window frame
        )
        pygame.display.set_caption(f"Exporting: {self.session_name}")
        
        # Minimize the window immediately (reduces GPU display overhead)
        pygame.display.iconify()
        
        # OpenGL setup
        bg = self.config.background_color
        glClearColor(bg[0], bg[1], bg[2], 1.0)
        glEnable(GL_DEPTH_TEST)
        
        # Fog for depth perception
        glEnable(GL_FOG)
        glFogi(GL_FOG_MODE, GL_EXP2)
        glFogfv(GL_FOG_COLOR, (bg[0], bg[1], bg[2], 1.0))
        glFogf(GL_FOG_DENSITY, self.config.fog_density)
        
        # Projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(75.0, self.width / self.height, 0.1, 10000.0)
        glMatrixMode(GL_MODELVIEW)
    
    def _render_frame(self, positions: np.ndarray, colors: np.ndarray):
        """Render a single frame to the OpenGL buffer (no display needed)."""
        from OpenGL.GL import (
            glClear, glEnable, glDisable, glBlendFunc, glPointSize,
            glEnableClientState, glDisableClientState, glVertexPointer,
            glColorPointer, glDrawArrays, glFinish,
            GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_POINT_SMOOTH,
            GL_BLEND, GL_SRC_ALPHA, GL_ONE, GL_VERTEX_ARRAY,
            GL_COLOR_ARRAY, GL_FLOAT, GL_POINTS
        )
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.camera.apply()
        
        glEnable(GL_POINT_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        glPointSize(self.config.point_size)
        
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, positions)
        glColorPointer(3, GL_FLOAT, 0, colors)
        glDrawArrays(GL_POINTS, 0, len(positions))
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
        
        glDisable(GL_BLEND)
        glDisable(GL_POINT_SMOOTH)
        
        # Ensure rendering is complete before reading pixels
        glFinish()
    
    def _capture_frame(self) -> bytes:
        """Capture the current OpenGL frame as raw RGB bytes."""
        from OpenGL.GL import glReadPixels, GL_RGB, GL_UNSIGNED_BYTE
        
        pixels = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        frame = np.frombuffer(pixels, dtype=np.uint8).reshape(self.height, self.width, 3)
        frame = np.flipud(frame)  # OpenGL has origin at bottom-left
        return frame.tobytes()
    
    def _get_ffmpeg_command(self) -> list:
        """Build FFmpeg command for encoding."""
        codec = self.config.codec
        crf = self.config.crf
        preset = self.config.encoding_preset
        
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.config.fps),
            "-i", "-",  # Read from pipe
        ]
        
        if codec == "h264":
            cmd.extend([
                "-c:v", "libx264",
                "-preset", preset,
                "-crf", str(crf),
                "-pix_fmt", "yuv420p",
                # Optimize for quality
                "-profile:v", "high",
                "-level", "4.2",
                # Better motion estimation
                "-x264-params", f"ref=4:bframes=3:b-adapt=2:direct=auto:me=umh:subme=8:trellis=2",
            ])
        elif codec == "h265":
            cmd.extend([
                "-c:v", "libx265",
                "-preset", preset,
                "-crf", str(crf),
                "-pix_fmt", "yuv420p",
                "-tag:v", "hvc1",  # Better compatibility
            ])
        elif codec == "vp9":
            cmd.extend([
                "-c:v", "libvpx-vp9",
                "-crf", str(crf),
                "-b:v", "0",
                "-pix_fmt", "yuv420p",
            ])
        
        # Faststart for web playback
        cmd.extend(["-movflags", "+faststart"])
        
        cmd.append(str(self.output_path))
        
        return cmd
    
    def export(self):
        """Export the video."""
        import pygame
        import time
        
        if not check_ffmpeg():
            print("[Export] Error: FFmpeg not found!")
            print("[Export] Install FFmpeg:")
            print("  macOS:   brew install ffmpeg")
            print("  Ubuntu:  sudo apt install ffmpeg")
            print("  Windows: Download from https://ffmpeg.org/download.html")
            return False
        
        print(f"\n{'='*60}")
        print(f"  N-BODY VIDEO EXPORT")
        print(f"{'='*60}")
        print(f"  Session:    {self.session_name}")
        print(f"  Bodies:     {self.metadata['num_bodies']:,}")
        print(f"  Frames:     {self.export_frames} ({self.start_frame}-{self.end_frame})")
        print(f"  Resolution: {self.width}x{self.height}")
        print(f"  FPS:        {self.config.fps}")
        print(f"  Quality:    {self.config.quality_preset} (CRF={self.config.crf})")
        print(f"  Codec:      {self.config.codec}")
        print(f"  Camera:     {self.config.camera_mode}")
        print(f"  Output:     {self.output_path}")
        print(f"{'='*60}\n")
        
        # Setup OpenGL
        self._setup_opengl()
        
        # Start FFmpeg process
        # Note: stderr=DEVNULL prevents pipe buffer from filling up and blocking FFmpeg
        ffmpeg_cmd = self._get_ffmpeg_command()
        ffmpeg_process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,  # Discard stderr to prevent blocking
        )
        
        start_time = time.time()
        
        # Maintain previous frame state for efficient delta decompression
        # Load first frame to initialize prev state
        prev_positions = None
        prev_colors = None
        
        # If starting from frame 0, we can load it first to initialize prev state
        if self.start_frame > 0:
            # Load the frame before start_frame to initialize prev state
            prev_positions, prev_colors = load_frame(self.rec_dir, self.start_frame - 1)
        
        try:
            for i, frame_idx in enumerate(range(self.start_frame, self.end_frame)):
                # Handle pygame events (allows window to be responsive)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("\n[Export] Cancelled by user")
                        ffmpeg_process.terminate()
                        pygame.quit()
                        return False
                
                # Load frame data (use previous frame for faster delta decompression)
                positions, colors = load_frame(self.rec_dir, frame_idx, prev_positions, prev_colors)
                
                # Store for next frame's delta decompression
                prev_positions = positions.copy()
                prev_colors = colors.copy()
                
                # Update camera
                self.camera.update(i, self.export_frames)
                
                # Render and capture
                self._render_frame(positions, colors)
                frame_data = self._capture_frame()
                
                # Send to FFmpeg
                ffmpeg_process.stdin.write(frame_data)
                
                # Progress update
                if (i + 1) % 10 == 0 or i == 0:
                    elapsed = time.time() - start_time
                    frames_done = i + 1
                    fps = frames_done / elapsed if elapsed > 0 else 0
                    eta = (self.export_frames - frames_done) / fps if fps > 0 else 0
                    pct = frames_done / self.export_frames * 100
                    
                    bar_width = 30
                    filled = int(bar_width * frames_done / self.export_frames)
                    bar = "█" * filled + "░" * (bar_width - filled)
                    
                    print(f"\r[{bar}] {pct:5.1f}% | {frames_done}/{self.export_frames} | "
                          f"{fps:.1f} fps | ETA: {format_time(eta)}    ", end="", flush=True)
            
            # Close FFmpeg stdin and wait for encoding to finish
            print(f"\n\n[Export] Finalizing video (encoding remaining frames)...")
            print("[Export] This may take several minutes for high-quality H.265...")
            
            finalize_start = time.time()
            ffmpeg_process.stdin.close()
            
            # Poll for completion with progress updates
            while ffmpeg_process.poll() is None:
                finalize_elapsed = time.time() - finalize_start
                print(f"\r[Export] Encoding... {format_time(finalize_elapsed)} elapsed    ", end="", flush=True)
                time.sleep(1)
            
            print(f"\r[Export] Encoding complete! ({format_time(time.time() - finalize_start)})    ")
            
            elapsed = time.time() - start_time
            
            # Check output
            if self.output_path.exists():
                file_size = self.output_path.stat().st_size
                duration = self.export_frames / self.config.fps
                bitrate = file_size * 8 / duration / 1_000_000 if duration > 0 else 0
                
                print(f"\n\n{'='*60}")
                print(f"  EXPORT COMPLETE")
                print(f"{'='*60}")
                print(f"  Output:   {self.output_path}")
                print(f"  Size:     {format_size(file_size)}")
                print(f"  Duration: {duration:.1f}s ({self.export_frames} frames @ {self.config.fps}fps)")
                print(f"  Bitrate:  {bitrate:.2f} Mbps")
                print(f"  Time:     {format_time(elapsed)}")
                print(f"{'='*60}\n")
                
                return True
            else:
                print(f"\n[Export] Error: FFmpeg failed to create output file")
                print(f"[Export] Try running with --quality balanced for faster encoding")
                return False
                
        except Exception as e:
            print(f"\n[Export] Error: {e}")
            ffmpeg_process.terminate()
            return False
        finally:
            pygame.quit()


# =============================================================================
# CLI
# =============================================================================

def list_recordings():
    """List available recordings."""
    recordings_dir = PROJECT_ROOT / "recordings"
    
    if not recordings_dir.exists():
        print("[Export] No recordings directory found")
        return
    
    sessions = [d for d in recordings_dir.iterdir() 
                if d.is_dir() and (d / "metadata.json").exists()]
    
    if not sessions:
        print("[Export] No recordings found")
        return
    
    print(f"\n{'='*70}")
    print("  AVAILABLE RECORDINGS")
    print(f"{'='*70}\n")
    
    for session_dir in sorted(sessions, key=lambda x: x.stat().st_mtime, reverse=True):
        metadata = load_metadata(session_dir)
        frame_count = get_frame_count(session_dir)
        
        # Check for existing export (in recordings/ folder)
        export_path = recordings_dir / f"{session_dir.name}.mp4"
        exported = "✓" if export_path.exists() else " "
        
        print(f"  [{exported}] {session_dir.name:30s} | {metadata['num_bodies']:>10,} bodies | {frame_count:>4} frames")
    
    print(f"\n  [✓] = Already exported to MP4")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Export N-body recording to optimized MP4 video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m tools.export galaxy_epic                    # Export with defaults
  python -m tools.export galaxy_epic --fps 60           # 60 FPS export  
  python -m tools.export galaxy_epic --quality high     # High quality
  python -m tools.export galaxy_epic --resolution 4k    # 4K resolution
  python -m tools.export galaxy_epic --camera cinematic # Cinematic camera

Quality Presets:
  fast      - Quick export, larger file
  balanced  - Good balance (default)
  high      - Best quality, slower export
  lossless  - Visually lossless

Camera Modes:
  fixed     - Static camera
  orbit     - Horizontal orbit (default)
  spiral    - Spiral motion
  zoom      - Zoom in/out cycle
  zoomout   - Constant zoom out
  zoomin    - Constant zoom in
  cinematic - Dramatic slow sweep
  flyby     - Dramatic flyby
  topdown   - Top-down rotating view
        """
    )
    
    parser.add_argument("session", nargs="?", help="Recording session name")
    parser.add_argument("--list", action="store_true", help="List available recordings")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode with prompts")
    
    # Video settings
    parser.add_argument("--fps", type=int, help="Output FPS (default: 30)")
    parser.add_argument("--resolution", type=str,
                        choices=list(RESOLUTION_PRESETS.keys()),
                        help="Output resolution (default: 1080p)")
    
    # Quality settings
    parser.add_argument("--quality", type=str,
                        choices=list(QUALITY_PRESETS.keys()),
                        help="Quality preset (default: balanced)")
    parser.add_argument("--crf", type=int, help="Override CRF value (0-51, lower=better)")
    parser.add_argument("--codec", type=str,
                        choices=["h264", "h265", "vp9"],
                        help="Video codec (default: h264)")
    
    # Camera settings
    parser.add_argument("--camera", type=str,
                        choices=["fixed", "orbit", "spiral", "zoom", "zoomout", "zoomin", "cinematic", "flyby", "topdown"],
                        help="Camera animation mode (default: orbit)")
    parser.add_argument("--camera-speed", type=float, default=0.3,
                        help="Camera rotation speed in degrees/frame (default: 0.3)")
    parser.add_argument("--camera-radius", type=float, default=800.0,
                        help="Camera distance from center / zoom level (default: 800.0, lower=closer)")
    parser.add_argument("--camera-angle", type=float, default=25.0,
                        help="Camera vertical angle (0=horizon, 90=top-down, default: 25.0)")
    parser.add_argument("--camera-theta", type=float, default=45.0,
                        help="Camera horizontal starting angle in degrees (default: 45.0)")
    
    # Rendering settings
    parser.add_argument("--point-size", type=float, default=1.5,
                        help="Point size for particles")
    
    # Frame range
    parser.add_argument("--start", type=int, help="Start frame")
    parser.add_argument("--end", type=int, help="End frame")
    
    # Output
    parser.add_argument("-o", "--output", type=str, help="Output file path")
    
    args = parser.parse_args()
    
    if args.list:
        list_recordings()
        return
    
    if not args.session:
        list_recordings()
        print("\nUsage: python -m tools.export <session_name> [options]")
        print("       python -m tools.export <session_name> -i    (interactive mode)")
        print("       python -m tools.export --help for more options")
        return
    
    # Check if recording exists
    rec_dir = get_recording_dir(args.session)
    if not rec_dir.exists() or not (rec_dir / "metadata.json").exists():
        print(f"[Export] Recording not found: {args.session}")
        list_recordings()
        return
    
    # Interactive mode if -i flag or no options provided
    use_interactive = args.interactive or (
        args.fps is None and args.resolution is None and 
        args.quality is None and args.camera is None
    )
    
    if use_interactive:
        config = interactive_export_config(args.session)
        if config is None:
            return
    else:
        # Build config from args
        config = ExportConfig()
        
    # Video settings
        config.fps = args.fps or 30
        config.resolution = RESOLUTION_PRESETS.get(args.resolution, (1920, 1080))
        
    # Quality settings
        quality_name = args.quality or "balanced"
        quality = QUALITY_PRESETS[quality_name]
        config.quality_preset = quality_name
        config.crf = args.crf if args.crf is not None else quality["crf"]
        config.encoding_preset = quality["encoding_preset"]
        config.codec = args.codec or "h264"
        
        # Camera settings
        config.camera_mode = args.camera or "orbit"
        config.camera_rotation_speed = args.camera_speed
        config.camera_radius = args.camera_radius
        config.camera_initial_phi = args.camera_angle
        config.camera_initial_theta = args.camera_theta
        
        # Rendering
        config.point_size = args.point_size
        
    # Frame range
        config.start_frame = args.start
        config.end_frame = args.end
        
    # Output
        config.output_path = args.output
    
    # Export
    try:
        exporter = VideoExporter(args.session, config)
        exporter.export()
    except FileNotFoundError as e:
        print(f"[Export] Error: {e}")
        list_recordings()
    except Exception as e:
        print(f"[Export] Error: {e}")
        raise


# Camera mode descriptions for interactive menu
CAMERA_MODES = {
    "orbit": "Horizontal orbit around simulation",
    "fixed": "Static camera position",
    "spiral": "Spiral motion with vertical oscillation",
    "zoom": "Zoom in/out cycle",
    "zoomout": "Constant zoom out (close → far)",
    "zoomin": "Constant zoom in (far → close)",
    "cinematic": "Dramatic slow sweep with zoom",
    "flyby": "Dramatic flyby past simulation",
    "topdown": "Top-down rotating view",
}


def interactive_export_config(session_name: str) -> Optional[ExportConfig]:
    """Interactive export configuration with prompts."""
    rec_dir = get_recording_dir(session_name)
    metadata = load_metadata(rec_dir)
    frame_count = get_frame_count(rec_dir)
    
    print(f"\n{'=' * 60}")
    print(f"  EXPORT: {session_name}")
    print(f"{'=' * 60}")
    print(f"  Bodies: {metadata.get('num_bodies', 'unknown'):,}")
    print(f"  Frames: {frame_count}")
    print(f"  Distribution: {metadata.get('distribution', 'unknown')}")
    
    config = ExportConfig()
    
    try:
        # Camera mode selection
        print(f"\n{'─' * 60}")
        print("  CAMERA MODE")
        print(f"{'─' * 60}")
        for i, (mode, desc) in enumerate(CAMERA_MODES.items()):
            marker = "→" if mode == "orbit" else " "
            print(f"  {marker} [{i}] {mode:12s} - {desc}")
        
        camera_input = input(f"\n  Select camera [0-{len(CAMERA_MODES)-1}] (Enter=orbit): ").strip()
        if camera_input:
            try:
                idx = int(camera_input)
                if 0 <= idx < len(CAMERA_MODES):
                    config.camera_mode = list(CAMERA_MODES.keys())[idx]
                    print(f"    → Camera: {config.camera_mode}")
            except ValueError:
                if camera_input in CAMERA_MODES:
                    config.camera_mode = camera_input
                    print(f"    → Camera: {config.camera_mode}")
        else:
            config.camera_mode = "orbit"
        
        # Camera angle selection
        print(f"\n{'─' * 60}")
        print("  CAMERA ANGLE")
        print(f"{'─' * 60}")
        print("  Vertical viewing angle:")
        print("    0°  - Horizon level (side view)")
        print("    25° - Slight elevation (default)")
        print("    45° - Diagonal view")
        print("    90° - Top-down view")
        angle_input = input(f"\n  Angle in degrees (Enter=25): ").strip()
        if angle_input:
            try:
                config.camera_initial_phi = float(angle_input)
                print(f"    → Angle: {config.camera_initial_phi}°")
            except ValueError:
                config.camera_initial_phi = 25.0
        else:
            config.camera_initial_phi = 25.0
        
        # Camera zoom level (radius)
        print(f"\n{'─' * 60}")
        print("  CAMERA ZOOM LEVEL")
        print(f"{'─' * 60}")
        print("  Distance from center (lower = closer, higher = farther):")
        print("    400  - Very close")
        print("    800  - Medium distance (default)")
        print("    1200 - Far away")
        print("    2000 - Very far")
        zoom_input = input(f"\n  Zoom level / radius (Enter=800): ").strip()
        if zoom_input:
            try:
                config.camera_radius = float(zoom_input)
                print(f"    → Zoom: {config.camera_radius}")
            except ValueError:
                config.camera_radius = 800.0
        else:
            config.camera_radius = 800.0
        
        # Camera orbit speed (if orbit mode)
        if config.camera_mode in ["orbit", "spiral", "zoom", "zoomout", "zoomin", "cinematic", "topdown"]:
            print(f"\n{'─' * 60}")
            print("  ORBIT ROTATION SPEED")
            print(f"{'─' * 60}")
            print("  How fast the camera rotates (degrees per frame):")
            print("    0.1  - Very slow")
            print("    0.3  - Normal speed (default)")
            print("    0.5  - Fast")
            print("    1.0  - Very fast")
            speed_input = input(f"\n  Rotation speed (Enter=0.3): ").strip()
            if speed_input:
                try:
                    config.camera_rotation_speed = float(speed_input)
                    print(f"    → Speed: {config.camera_rotation_speed}°/frame")
                except ValueError:
                    config.camera_rotation_speed = 0.3
            else:
                config.camera_rotation_speed = 0.3
        
        # Camera horizontal starting angle
        print(f"\n{'─' * 60}")
        print("  CAMERA STARTING POSITION")
        print(f"{'─' * 60}")
        print("  Horizontal starting angle (where camera starts orbiting from):")
        print("    0°   - Front view")
        print("    45°  - Diagonal front (default)")
        print("    90°  - Side view")
        print("    180° - Back view")
        theta_input = input(f"\n  Starting angle in degrees (Enter=45): ").strip()
        if theta_input:
            try:
                config.camera_initial_theta = float(theta_input)
                print(f"    → Starting angle: {config.camera_initial_theta}°")
            except ValueError:
                config.camera_initial_theta = 45.0
        else:
            config.camera_initial_theta = 45.0
        
        # Point size (rendering)
        print(f"\n{'─' * 60}")
        print("  POINT SIZE")
        print(f"{'─' * 60}")
        print("  Size of particles in the render:")
        print("    1.0  - Small")
        print("    1.5  - Medium (default)")
        print("    2.0  - Large")
        print("    3.0  - Very large")
        point_input = input(f"\n  Point size (Enter=1.5): ").strip()
        if point_input:
            try:
                config.point_size = float(point_input)
                print(f"    → Point size: {config.point_size}")
            except ValueError:
                config.point_size = 1.5
        else:
            config.point_size = 1.5
        
        # FPS selection
        print(f"\n{'─' * 60}")
        print("  FPS")
        print(f"{'─' * 60}")
        print("  Common: 24 (cinema), 30 (standard), 60 (smooth), 120 (high)")
        fps_input = input(f"  FPS (Enter=30): ").strip()
        if fps_input:
            try:
                config.fps = int(fps_input)
                print(f"    → FPS: {config.fps}")
            except ValueError:
                config.fps = 30
        else:
            config.fps = 30
        
        # Resolution selection
        print(f"\n{'─' * 60}")
        print("  RESOLUTION")
        print(f"{'─' * 60}")
        res_list = list(RESOLUTION_PRESETS.items())
        for i, (name, (w, h)) in enumerate(res_list):
            marker = "→" if name == "1080p" else " "
            print(f"  {marker} [{i}] {name:10s} ({w}x{h})")
        
        res_input = input(f"\n  Select resolution [0-{len(res_list)-1}] (Enter=1080p): ").strip()
        if res_input:
            try:
                idx = int(res_input)
                if 0 <= idx < len(res_list):
                    res_name = res_list[idx][0]
                    config.resolution = RESOLUTION_PRESETS[res_name]
                    print(f"    → Resolution: {res_name}")
            except ValueError:
                if res_input in RESOLUTION_PRESETS:
                    config.resolution = RESOLUTION_PRESETS[res_input]
        else:
            config.resolution = (1920, 1080)
        
        # Quality selection
        print(f"\n{'─' * 60}")
        print("  QUALITY")
        print(f"{'─' * 60}")
        quality_list = list(QUALITY_PRESETS.items())
        for i, (name, q) in enumerate(quality_list):
            marker = "→" if name == "balanced" else " "
            print(f"  {marker} [{i}] {name:10s} - {q['description']}")
        
        quality_input = input(f"\n  Select quality [0-{len(quality_list)-1}] (Enter=balanced): ").strip()
        if quality_input:
            try:
                idx = int(quality_input)
                if 0 <= idx < len(quality_list):
                    quality_name = quality_list[idx][0]
                    quality = QUALITY_PRESETS[quality_name]
                    config.quality_preset = quality_name
                    config.crf = quality["crf"]
                    config.encoding_preset = quality["encoding_preset"]
                    print(f"    → Quality: {quality_name}")
            except ValueError:
                pass
        else:
            config.quality_preset = "balanced"
            config.crf = 23
            config.encoding_preset = "medium"
        
        # Codec selection
        print(f"\n{'─' * 60}")
        print("  CODEC")
        print(f"{'─' * 60}")
        print("  → [0] h264    - Most compatible (default)")
        print("    [1] h265    - Better compression, less compatible")
        print("    [2] vp9     - Open format, good compression")
        
        codec_input = input(f"\n  Select codec [0-2] (Enter=h264): ").strip()
        codecs = ["h264", "h265", "vp9"]
        if codec_input:
            try:
                idx = int(codec_input)
                if 0 <= idx < 3:
                    config.codec = codecs[idx]
                    print(f"    → Codec: {config.codec}")
            except ValueError:
                config.codec = "h264"
        else:
            config.codec = "h264"
        
        # Summary
        print(f"\n{'=' * 60}")
        print("  EXPORT SETTINGS")
        print(f"{'=' * 60}")
        print(f"  Session:    {session_name}")
        print(f"  Camera:     {config.camera_mode}")
        print(f"    Angle:    {config.camera_initial_phi}° (vertical)")
        print(f"    Start:    {config.camera_initial_theta}° (horizontal)")
        print(f"    Zoom:     {config.camera_radius}")
        if config.camera_mode != "fixed":
            print(f"    Speed:    {config.camera_rotation_speed}°/frame")
        print(f"  FPS:        {config.fps}")
        print(f"  Resolution: {config.resolution[0]}x{config.resolution[1]}")
        print(f"  Quality:    {config.quality_preset}")
        print(f"  Codec:      {config.codec}")
        print(f"  Point size: {config.point_size}")
        
        duration = frame_count / config.fps
        print(f"  Duration:   {duration:.1f}s ({frame_count} frames)")
        
        confirm = input(f"\n  Start export? [Y/n]: ").strip().lower()
        if confirm in ['', 'y', 'yes']:
            return config
        else:
            print("  Cancelled.")
            return None
            
    except KeyboardInterrupt:
        print("\n  Cancelled.")
        return None


if __name__ == "__main__":
    main()

