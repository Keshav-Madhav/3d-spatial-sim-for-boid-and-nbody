"""
N-Body Recording Playback
=========================

Plays back recorded simulation frames at any FPS.
For video export, use: python -m tools.export

Usage:
    python -m tools.playback <session_name>              # Playback at default FPS
    python -m tools.playback <session_name> --fps 60    # Custom FPS
    python -m tools.playback <session_name> --loop      # Loop playback

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
import threading
from collections import deque
from pathlib import Path

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.arrays import vbo

# Get project root (parent of tools/)
PROJECT_ROOT = Path(__file__).parent.parent

# Import shared functions from record module
from tools.record import get_recording_dir, load_metadata, get_completed_frames as get_frame_count, load_frame


class PlaybackCamera:
    """Simple orbital camera for playback with full 360° rotation."""
    
    def __init__(self):
        self.radius = 800.0
        self.theta = 45.0
        self.phi = 25.0
        self.target = np.array([0.0, 0.0, 0.0])
        self.min_radius = -3000.0
        self.max_radius = 3000.0
    
    def rotate(self, d_theta: float, d_phi: float):
        self.theta = (self.theta + d_theta) % 360
        self.phi = (self.phi + d_phi) % 360
    
    def zoom(self, delta: float):
        """Zoom by delta amount, allowing negative radius (zooming past center)."""
        self.radius = max(self.min_radius, min(self.max_radius, self.radius + delta))
    
    def get_position(self) -> np.ndarray:
        import math
        theta_rad = math.radians(self.theta)
        phi_rad = math.radians(self.phi)
        x = self.radius * math.cos(phi_rad) * math.cos(theta_rad)
        y = self.radius * math.sin(phi_rad)
        z = self.radius * math.cos(phi_rad) * math.sin(theta_rad)
        return np.array([x, y, z])
    
    def get_direction(self) -> np.ndarray:
        """Get the direction vector from target to camera."""
        import math
        theta_rad = math.radians(self.theta)
        phi_rad = math.radians(self.phi)
        x = math.cos(phi_rad) * math.cos(theta_rad)
        y = math.sin(phi_rad)
        z = math.cos(phi_rad) * math.sin(theta_rad)
        return np.array([x, y, z])
    
    def get_up_vector(self) -> tuple:
        import math
        phi_rad = math.radians(self.phi)
        if math.cos(phi_rad) >= 0:
            return (0, 1, 0)
        else:
            return (0, -1, 0)
    
    def apply(self):
        """Apply camera transformation, handling negative radius (zooming past center)."""
        pos = self.get_position()
        up = self.get_up_vector()
        glLoadIdentity()
        
        # If radius is negative, we're inside the simulation - adjust look_at
        if self.radius >= 0:
            look_at = self.target
        else:
            # When inside, look away from center
            direction = self.get_direction()
            look_at = pos - direction * 10
        
        gluLookAt(
            pos[0], pos[1], pos[2],
            look_at[0], look_at[1], look_at[2],
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
        
        # Frame buffer for on-demand loading with background preloading
        self.frame_cache = {}  # frame_idx -> (positions, colors)
        self.cache_lock = threading.Lock()
        self.cache_size = 50  # Keep up to 50 frames in cache
        self.preload_thread = None
        self.preload_running = False
        
        if self.preload_all:
            print("[Playback] Preloading all frames...")
            for i in range(self.frame_count):
                self.frames.append(load_frame(self.rec_dir, i))
                if (i + 1) % 20 == 0:
                    print(f"  Loaded {i+1}/{self.frame_count}")
        else:
            # Preload first batch of frames for smooth start
            print("[Playback] Preloading initial frames...")
            preload_count = min(30, self.frame_count)
            for i in range(preload_count):
                self.frames.append(load_frame(self.rec_dir, i))
            print(f"  Preloaded {preload_count} frames")
            
            # Start background preloading thread
            self.preload_running = True
            self.preload_thread = threading.Thread(target=self._preload_worker, daemon=True)
            self.preload_thread.start()
        
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
    
    def _preload_worker(self):
        """Background thread that preloads frames ahead of current playback."""
        prev_positions = None
        prev_colors = None
        
        while self.preload_running:
            try:
                # Get current frame and preload ahead
                current = self.current_frame
                
                # Preload frames ahead (next 30 frames for smooth playback)
                for offset in range(1, 31):
                    frame_idx = current + offset
                    
                    # Stop if we've reached the end
                    if frame_idx >= self.frame_count:
                        break
                    
                    # Check if already cached
                    with self.cache_lock:
                        if frame_idx in self.frame_cache:
                            # Update prev for next frame
                            prev_positions, prev_colors = self.frame_cache[frame_idx]
                            continue
                    
                    # Load frame sequentially (needed for delta compression)
                    try:
                        # Use previous frame if available for faster delta decompression
                        if prev_positions is not None and prev_colors is not None:
                            frame_data = load_frame(self.rec_dir, frame_idx, prev_positions, prev_colors)
                        else:
                            # Load from beginning if we don't have previous
                            frame_data = load_frame(self.rec_dir, frame_idx)
                        
                        prev_positions, prev_colors = frame_data
                        
                        with self.cache_lock:
                            # Limit cache size - remove oldest if needed
                            if len(self.frame_cache) >= self.cache_size:
                                # Remove oldest frame (not current or recent)
                                oldest = min((k for k in self.frame_cache.keys() 
                                             if k < current - 5), default=None)
                                if oldest is not None:
                                    del self.frame_cache[oldest]
                            
                            self.frame_cache[frame_idx] = frame_data
                    except Exception as e:
                        # Frame might not exist yet, skip
                        prev_positions = None
                        prev_colors = None
                        break  # Stop preloading if we hit an error
                
                # Small sleep to avoid hogging CPU
                threading.Event().wait(0.05)  # Check more frequently
                
            except Exception:
                # Continue on error
                prev_positions = None
                prev_colors = None
                threading.Event().wait(0.1)
    
    def _get_frame_data(self, idx: int) -> tuple:
        """Get frame data, using cache if available."""
        if self.preload_all:
            return self.frames[idx]
        
        # Check cache first
        with self.cache_lock:
            if idx in self.frame_cache:
                return self.frame_cache[idx]
            
            # Try to get previous frame from cache for faster delta decompression
            prev_positions = None
            prev_colors = None
            if idx > 0 and (idx - 1) in self.frame_cache:
                prev_positions, prev_colors = self.frame_cache[idx - 1]
        
        # Not in cache, load directly (use cached previous frame if available for speed)
        frame_data = load_frame(self.rec_dir, idx, prev_positions, prev_colors)
        
        # Cache it for future use
        with self.cache_lock:
            if len(self.frame_cache) >= self.cache_size:
                # Remove oldest frame (keep recent frames)
                oldest = min((k for k in self.frame_cache.keys() if k != idx), default=None)
                if oldest is not None:
                    del self.frame_cache[oldest]
            self.frame_cache[idx] = frame_data
        
        return frame_data
    
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
        
        try:
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
        finally:
            # Stop preload thread
            self.preload_running = False
            if self.preload_thread:
                self.preload_thread.join(timeout=2.0)
        
        pygame.quit()




def main():
    parser = argparse.ArgumentParser(description="N-Body recording playback")
    parser.add_argument("session", nargs="?", default="galaxy_1m", help="Recording session name")
    parser.add_argument("--fps", type=int, default=30, help="Playback FPS")
    parser.add_argument("--loop", action="store_true", help="Loop playback")
    parser.add_argument("--export", action="store_true", help="(Deprecated) Use: python -m tools.export")
    args = parser.parse_args()
    
    if args.export:
        print("[Playback] Export moved to dedicated tool!")
        print(f"[Playback] Use: python -m tools.export {args.session}")
        print("[Playback] Run: python -m tools.export --help for options")
        return
    
    app = PlaybackApp(args.session, fps=args.fps, loop=args.loop)
    app.run()


if __name__ == "__main__":
    main()

