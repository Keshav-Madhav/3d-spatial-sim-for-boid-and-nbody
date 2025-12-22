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

def get_recording_dir(session_name: str) -> Path:
    """Get the directory for a recording session."""
    return PROJECT_ROOT / "recordings" / session_name


def load_metadata(rec_dir: Path) -> dict:
    """Load recording metadata."""
    with open(rec_dir / "metadata.json", "r") as f:
        return json.load(f)


def get_frame_count(rec_dir: Path) -> int:
    """Count available frames."""
    count = 0
    while (rec_dir / f"frame_{count:04d}.npz").exists():
        count += 1
    return count


def load_frame(rec_dir: Path, frame_idx: int) -> tuple:
    """Load a single frame from disk."""
    with np.load(rec_dir / f"frame_{frame_idx:04d}.npz") as data:
        return data["positions"].copy(), data["colors"].copy()


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
        elif mode == "cinematic":
            # Dramatic slow sweep
            self.theta = self.config.camera_initial_theta + frame_idx * speed * 0.3
            self.phi = self.config.camera_initial_phi + 15 * np.sin(t * np.pi)
            self.radius = self.config.camera_radius * (1.0 - 0.2 * t)
    
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
        
        # Output path
        if config.output_path:
            self.output_path = Path(config.output_path)
        else:
            self.output_path = self.rec_dir / f"{session_name}.mp4"
        
        self.width, self.height = config.resolution
        self.camera = ExportCamera(config)
        
        # Temp directory for frame export
        self.temp_dir = None
        self.use_pipe = True  # Pipe frames directly to FFmpeg
    
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
        
        try:
            for i, frame_idx in enumerate(range(self.start_frame, self.end_frame)):
                # Handle pygame events (allows window to be responsive)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("\n[Export] Cancelled by user")
                        ffmpeg_process.terminate()
                        pygame.quit()
                        return False
                
                # Load frame data
                positions, colors = load_frame(self.rec_dir, frame_idx)
                
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
        
        # Check for existing export
        export_path = session_dir / f"{session_dir.name}.mp4"
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
  cinematic - Dramatic slow sweep
        """
    )
    
    parser.add_argument("session", nargs="?", help="Recording session name")
    parser.add_argument("--list", action="store_true", help="List available recordings")
    
    # Video settings
    parser.add_argument("--fps", type=int, default=30, help="Output FPS (default: 30)")
    parser.add_argument("--resolution", type=str, default="1080p",
                        choices=list(RESOLUTION_PRESETS.keys()),
                        help="Output resolution (default: 1080p)")
    
    # Quality settings
    parser.add_argument("--quality", type=str, default="balanced",
                        choices=list(QUALITY_PRESETS.keys()),
                        help="Quality preset (default: balanced)")
    parser.add_argument("--crf", type=int, help="Override CRF value (0-51, lower=better)")
    parser.add_argument("--codec", type=str, default="h264",
                        choices=["h264", "h265", "vp9"],
                        help="Video codec (default: h264)")
    
    # Camera settings
    parser.add_argument("--camera", type=str, default="orbit",
                        choices=["fixed", "orbit", "spiral", "zoom", "cinematic"],
                        help="Camera animation mode (default: orbit)")
    parser.add_argument("--camera-speed", type=float, default=0.3,
                        help="Camera rotation speed in degrees/frame")
    parser.add_argument("--camera-radius", type=float, default=800.0,
                        help="Camera distance from center")
    
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
        print("Usage: python -m tools.export <session_name> [options]")
        print("       python -m tools.export --help for more options")
        return
    
    # Build config
    config = ExportConfig()
    
    # Video settings
    config.fps = args.fps
    config.resolution = RESOLUTION_PRESETS[args.resolution]
    
    # Quality settings
    quality = QUALITY_PRESETS[args.quality]
    config.quality_preset = args.quality
    config.crf = args.crf if args.crf is not None else quality["crf"]
    config.encoding_preset = quality["encoding_preset"]
    config.codec = args.codec
    
    # Camera settings
    config.camera_mode = args.camera
    config.camera_rotation_speed = args.camera_speed
    config.camera_radius = args.camera_radius
    
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


if __name__ == "__main__":
    main()

