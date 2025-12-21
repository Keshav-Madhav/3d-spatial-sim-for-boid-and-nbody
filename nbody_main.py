"""
N-Body Gravitational Simulation
===============================

A real-time gravitational simulation with 1,000,000 bodies using
Barnes-Hut octree algorithm for O(n log n) performance.

Controls:
    - W/S: Rotate camera up/down
    - A/D: Rotate camera left/right
    - Q/E: Zoom in/out
    - Mouse drag: Rotate camera
    - Mouse wheel: Zoom
    - SPACE: Pause/Resume simulation
    - R: Reset simulation
    - H: Toggle help text
    - ESC: Quit
"""

# Import directly to avoid triggering core/__init__.py which imports boids
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

from config import nbody as config
from core.nbody_camera import NBodyCamera
from core.nbody_input_handler import NBodyInputHandler
from rendering import TextRenderer
from rendering.nbody_grid import NBodyGrid
from nbody import NBodySimulation


class NBodyApplication:
    """Main application for N-body gravitational simulation."""
    
    def __init__(self):
        pygame.init()
        
        # Use default OpenGL context (macOS compatibility)
        pygame.display.set_mode(
            (config.WINDOW["width"], config.WINDOW["height"]),
            DOUBLEBUF | OPENGL
        )
        pygame.display.set_caption(config.WINDOW["title"])
        
        # Core components
        self.camera = NBodyCamera()
        self.input_handler = NBodyInputHandler(self.camera)
        
        # Rendering components
        self.grid = NBodyGrid()
        self.text_renderer = TextRenderer()
        
        # N-body simulation
        print("[App] Initializing N-body simulation...")
        self.simulation = NBodySimulation(num_bodies=config.NBODY["count"])
        
        # State
        self.clock = pygame.time.Clock()
        self.running = True
        self.fps = 0
        self.paused = False
        self.show_help = True
        self.frame_count = 0
        
        self._setup_gl()
        print("[App] Ready!")
    
    def _setup_gl(self):
        """Initialize OpenGL settings."""
        glClearColor(*config.COLORS["background"])
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        
        # Enable fog for depth perception
        glEnable(GL_FOG)
        glFogi(GL_FOG_MODE, GL_EXP2)
        glFogfv(GL_FOG_COLOR, config.COLORS["background"])
        glFogf(GL_FOG_DENSITY, 0.0003)
        
        # Setup projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(
            config.CAMERA["fov"],
            config.WINDOW["width"] / config.WINDOW["height"],
            config.CAMERA["near_clip"],
            config.CAMERA["far_clip"]
        )
        glMatrixMode(GL_MODELVIEW)
    
    def _handle_events(self):
        """Process all pending pygame events."""
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    self.running = False
                elif event.key == K_SPACE:
                    self.paused = not self.paused
                    print(f"[App] {'Paused' if self.paused else 'Running'}")
                elif event.key == K_h:
                    self.show_help = not self.show_help
                elif event.key == K_r:
                    # Reset simulation
                    print("[App] Resetting simulation...")
                    self.simulation = NBodySimulation(num_bodies=config.NBODY["count"])
            else:
                if not self.input_handler.handle_event(event):
                    self.running = False
    
    def _update(self, dt: float):
        """Update simulation state."""
        dt = min(dt, 0.05)
        
        self.input_handler.handle_continuous_input(dt)
        self.camera.update(dt)
        
        if not self.paused:
            self.simulation.update(dt)
    
    def _render(self):
        """Render the scene."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.camera.apply()
        
        # Draw boundary grid
        self.grid.draw(self.camera)
        
        # Draw N-body simulation
        cam_pos = self.camera.get_position()
        cam_forward, cam_right, cam_up = self.camera.get_camera_axes()
        aspect = config.WINDOW["width"] / config.WINDOW["height"]
        
        self.simulation.draw(
            cam_pos, cam_forward, cam_right, cam_up,
            config.CAMERA["fov"], aspect
        )
        
        # Draw HUD
        screen_size = (config.WINDOW["width"], config.WINDOW["height"])
        visible = self.simulation._visible_count
        total = self.simulation.num_bodies
        
        status = "PAUSED" if self.paused else "RUNNING"
        self.text_renderer.draw_text(
            f"Bodies: {visible:,}/{total:,}  |  FPS: {self.fps:.0f}  |  {status}",
            10, 10, screen_size
        )
        self.text_renderer.draw_text(
            f"Tree nodes: {self.simulation._num_tree_nodes:,}  |  θ={config.NBODY['theta']}",
            10, 35, screen_size
        )
        
        if self.show_help:
            self.text_renderer.draw_text(
                "WASD: Rotate | QE: Zoom | SPACE: Pause | R: Reset | H: Toggle help",
                10, 60, screen_size
            )
        
        pygame.display.flip()
    
    def run(self):
        """Main application loop."""
        print("[App] Starting main loop...")
        
        while self.running:
            dt = self.clock.tick() / 1000.0  # Uncapped FPS
            self.fps = self.clock.get_fps()
            self.frame_count += 1
            
            self._handle_events()
            self._update(dt)
            self._render()
        
        pygame.quit()
        print("[App] Shutdown complete")


def main():
    app = NBodyApplication()
    app.run()


if __name__ == "__main__":
    main()

