"""
N-Body Recording Playback
=========================

Plays back recorded simulation frames at any FPS.
Can also export to video.

Usage:
    python -m tools.playback <session_name>              # Playback at default FPS
    python -m tools.playback <session_name> --fps 60    # Custom FPS
    python -m tools.playback <session_name> --loop      # Loop playback
    python -m tools.playback <session_name> --export    # Export to MP4

Controls during playback:
    Mouse drag  - Rotate camera (full 360°)
    Scroll      - Zoom in/out
    WASD        - Rotate camera
    Q/E         - Zoom in/out
    SPACE       - Pause/Resume
    LEFT/RIGHT  - Step frame
    UP/DOWN     - Adjust playback speed
    R           - Restart from beginning
    L           - Toggle loop mode
    F           - Toggle fullscreen
    ESC         - Quit
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.arrays import vbo

# Get project root (parent of tools/)
PROJECT_ROOT = Path(__file__).parent.parent


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
    data = np.load(rec_dir / f"frame_{frame_idx:04d}.npz")
    return data["positions"], data["colors"]


class PlaybackCamera:
    """Simple orbital camera for playback with full 360° rotation."""
    
    def __init__(self):
        self.radius = 800.0
        self.theta = 45.0
        self.phi = 25.0
        self.target = np.array([0.0, 0.0, 0.0])
    
    def rotate(self, d_theta: float, d_phi: float):
        self.theta = (self.theta + d_theta) % 360
        self.phi = (self.phi + d_phi) % 360
    
    def zoom(self, delta: float):
        self.radius = max(10, self.radius + delta)
    
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
        pos = self.get_position()
        up = self.get_up_vector()
        glLoadIdentity()
        gluLookAt(
            pos[0], pos[1], pos[2],
            self.target[0], self.target[1], self.target[2],
            up[0], up[1], up[2]
        )


class PlaybackApp:
    """Playback application for recorded N-body simulations."""
    
    def __init__(self, session_name: str, fps: int = 30, loop: bool = False):
        self.session_name = session_name
        self.target_fps = fps
        self.loop = loop
        
        self.rec_dir = get_recording_dir(session_name)
        if not self.rec_dir.exists():
            raise FileNotFoundError(f"Recording not found: {session_name}")
        
        self.metadata = load_metadata(self.rec_dir)
        self.frame_count = get_frame_count(self.rec_dir)
        
        if self.frame_count == 0:
            raise ValueError(f"No frames found in recording: {session_name}")
        
        print(f"[Playback] Loading: {session_name}")
        print(f"[Playback] Bodies: {self.metadata['num_bodies']:,}")
        print(f"[Playback] Frames: {self.frame_count}")
        print(f"[Playback] Theta: {self.metadata['theta']}")
        
        self.frames = []
        self.preload_all = self.frame_count <= 200
        
        if self.preload_all:
            print("[Playback] Preloading frames...")
            for i in range(self.frame_count):
                self.frames.append(load_frame(self.rec_dir, i))
                if (i + 1) % 20 == 0:
                    print(f"  Loaded {i+1}/{self.frame_count}")
        
        pygame.init()
        
        display_info = pygame.display.Info()
        self.width = display_info.current_w
        self.height = display_info.current_h
        
        pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL | FULLSCREEN)
        pygame.display.set_caption(f"N-Body Playback: {session_name}")
        
        self.camera = PlaybackCamera()
        self.clock = pygame.time.Clock()
        
        self.mouse_dragging = False
        self.last_mouse_pos = (0, 0)
        self.mouse_sensitivity = 0.3
        
        self.current_frame = 0
        self.playing = True
        self.speed = 1.0
        self.running = True
        
        self._vbo_positions = None
        self._vbo_colors = None
        self._vbos_initialized = False
        
        self._setup_gl()
        
        pygame.font.init()
        self.font = pygame.font.SysFont("monospace", 18)
    
    def _setup_gl(self):
        glClearColor(0.0, 0.0, 0.02, 1.0)
        glEnable(GL_DEPTH_TEST)
        
        glEnable(GL_FOG)
        glFogi(GL_FOG_MODE, GL_EXP2)
        glFogfv(GL_FOG_COLOR, (0.0, 0.0, 0.02, 1.0))
        glFogf(GL_FOG_DENSITY, 0.0003)
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(75.0, self.width / self.height, 0.1, 10000.0)
        glMatrixMode(GL_MODELVIEW)
    
    def _init_vbos(self, positions, colors):
        if self._vbos_initialized:
            return
        
        try:
            self._vbo_positions = vbo.VBO(positions, usage=GL_DYNAMIC_DRAW)
            self._vbo_colors = vbo.VBO(colors, usage=GL_DYNAMIC_DRAW)
            self._vbos_initialized = True
        except Exception as e:
            print(f"[Playback] VBO init failed: {e}")
    
    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    self.running = False
                elif event.key == K_SPACE:
                    self.playing = not self.playing
                elif event.key == K_LEFT:
                    self.current_frame = max(0, self.current_frame - 1)
                elif event.key == K_RIGHT:
                    self.current_frame = min(self.frame_count - 1, self.current_frame + 1)
                elif event.key == K_UP:
                    self.speed = min(4.0, self.speed * 1.5)
                elif event.key == K_DOWN:
                    self.speed = max(0.1, self.speed / 1.5)
                elif event.key == K_r:
                    self.current_frame = 0
                elif event.key == K_l:
                    self.loop = not self.loop
                elif event.key == K_f:
                    pygame.display.toggle_fullscreen()
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.mouse_dragging = True
                    self.last_mouse_pos = pygame.mouse.get_pos()
            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_dragging = False
            elif event.type == MOUSEMOTION:
                if self.mouse_dragging:
                    current_pos = pygame.mouse.get_pos()
                    dx = current_pos[0] - self.last_mouse_pos[0]
                    dy = current_pos[1] - self.last_mouse_pos[1]
                    self.camera.rotate(dx * self.mouse_sensitivity, -dy * self.mouse_sensitivity)
                    self.last_mouse_pos = current_pos
            elif event.type == MOUSEWHEEL:
                self.camera.zoom(-event.y * 50)
    
    def _handle_continuous_input(self, dt: float):
        keys = pygame.key.get_pressed()
        rot_speed = 60.0 * dt
        zoom_speed = 100.0 * dt
        
        if keys[K_a]:
            self.camera.rotate(-rot_speed, 0)
        if keys[K_d]:
            self.camera.rotate(rot_speed, 0)
        if keys[K_w]:
            self.camera.rotate(0, rot_speed)
        if keys[K_s]:
            self.camera.rotate(0, -rot_speed)
        if keys[K_q]:
            self.camera.zoom(-zoom_speed)
        if keys[K_e]:
            self.camera.zoom(zoom_speed)
    
    def _get_frame_data(self, idx: int) -> tuple:
        if self.preload_all:
            return self.frames[idx]
        else:
            return load_frame(self.rec_dir, idx)
    
    def _render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.camera.apply()
        
        positions, colors = self._get_frame_data(self.current_frame)
        
        if not self._vbos_initialized:
            self._init_vbos(positions, colors)
        
        glEnable(GL_POINT_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        glPointSize(1.5)
        
        if self._vbos_initialized:
            self._vbo_positions.set_array(positions)
            self._vbo_colors.set_array(colors)
            
            self._vbo_positions.bind()
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, None)
            
            self._vbo_colors.bind()
            glEnableClientState(GL_COLOR_ARRAY)
            glColorPointer(3, GL_FLOAT, 0, None)
            
            glDrawArrays(GL_POINTS, 0, len(positions))
            
            self._vbo_positions.unbind()
            self._vbo_colors.unbind()
            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_COLOR_ARRAY)
        else:
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_COLOR_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, positions)
            glColorPointer(3, GL_FLOAT, 0, colors)
            glDrawArrays(GL_POINTS, 0, len(positions))
            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_COLOR_ARRAY)
        
        glDisable(GL_BLEND)
        glDisable(GL_POINT_SMOOTH)
        
        self._draw_hud()
        
        pygame.display.flip()
    
    def _draw_hud(self):
        fps = self.clock.get_fps()
        status = "▶" if self.playing else "⏸"
        loop_status = "🔁" if self.loop else ""
        
        text = f"{status} Frame {self.current_frame+1}/{self.frame_count} | Speed: {self.speed:.1f}x | FPS: {fps:.0f} {loop_status}"
        
        text_surface = self.font.render(text, True, (200, 200, 220))
        text_data = pygame.image.tostring(text_surface, "RGBA", True)
        w, h = text_surface.get_size()
        
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, 0, self.height, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glRasterPos2f(10, self.height - 25)
        glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, text_data)
        glDisable(GL_BLEND)
        
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        
        if self.current_frame == 0 and self.playing:
            hint = "Mouse: Rotate | Scroll: Zoom | SPACE: Pause | ←→: Frame | F: Fullscreen | ESC: Quit"
            hint_surface = self.font.render(hint, True, (150, 150, 170))
            hint_data = pygame.image.tostring(hint_surface, "RGBA", True)
            hw, hh = hint_surface.get_size()
            
            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            glOrtho(0, self.width, 0, self.height, -1, 1)
            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            glLoadIdentity()
            
            glEnable(GL_BLEND)
            glRasterPos2f(10, self.height - 50)
            glDrawPixels(hw, hh, GL_RGBA, GL_UNSIGNED_BYTE, hint_data)
            glDisable(GL_BLEND)
            
            glPopMatrix()
            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
    
    def run(self):
        print(f"\n[Playback] Starting at {self.target_fps} FPS")
        print("[Playback] Controls: SPACE=pause, ←→=frame, ↑↓=speed, WASD/QE=camera, ESC=quit\n")
        
        frame_accumulator = 0.0
        
        while self.running:
            dt = self.clock.tick(60) / 1000.0
            
            self._handle_events()
            self._handle_continuous_input(dt)
            
            if self.playing:
                frame_accumulator += dt * self.target_fps * self.speed
                
                while frame_accumulator >= 1.0:
                    frame_accumulator -= 1.0
                    self.current_frame += 1
                    
                    if self.current_frame >= self.frame_count:
                        if self.loop:
                            self.current_frame = 0
                        else:
                            self.current_frame = self.frame_count - 1
                            self.playing = False
            
            self._render()
        
        pygame.quit()


def export_video(session_name: str, fps: int = 30, output_path: str = None):
    """Export recording to MP4 video."""
    try:
        import cv2
    except ImportError:
        print("[Export] Error: opencv-python required for video export")
        print("[Export] Install with: pip install opencv-python")
        return
    
    rec_dir = get_recording_dir(session_name)
    metadata = load_metadata(rec_dir)
    frame_count = get_frame_count(rec_dir)
    
    if output_path is None:
        output_path = str(rec_dir / f"{session_name}.mp4")
    
    print(f"[Export] Exporting {frame_count} frames to {output_path}")
    
    pygame.init()
    width, height = 1920, 1080
    pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    glClearColor(0.0, 0.0, 0.02, 1.0)
    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(75.0, width / height, 0.1, 10000.0)
    glMatrixMode(GL_MODELVIEW)
    
    camera = PlaybackCamera()
    
    for i in range(frame_count):
        positions, colors = load_frame(rec_dir, i)
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        camera.apply()
        
        glEnable(GL_POINT_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        glPointSize(1.5)
        
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, positions)
        glColorPointer(3, GL_FLOAT, 0, colors)
        glDrawArrays(GL_POINTS, 0, len(positions))
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
        
        glDisable(GL_BLEND)
        
        pygame.display.flip()
        
        pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        frame = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3)
        frame = np.flipud(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        video.write(frame)
        
        if (i + 1) % 10 == 0:
            print(f"  Exported {i+1}/{frame_count}")
        
        camera.rotate(0.5, 0)
    
    video.release()
    pygame.quit()
    
    print(f"[Export] ✓ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="N-Body recording playback")
    parser.add_argument("session", nargs="?", default="galaxy_1m", help="Recording session name")
    parser.add_argument("--fps", type=int, default=30, help="Playback FPS")
    parser.add_argument("--loop", action="store_true", help="Loop playback")
    parser.add_argument("--export", action="store_true", help="Export to MP4")
    args = parser.parse_args()
    
    if args.export:
        export_video(args.session, args.fps)
    else:
        app = PlaybackApp(args.session, fps=args.fps, loop=args.loop)
        app.run()


if __name__ == "__main__":
    main()

