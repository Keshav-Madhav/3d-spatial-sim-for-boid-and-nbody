"""Main application class that ties everything together."""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

from config import boids as config
from .camera import Camera
from .input_handler import InputHandler
from rendering import Grid, TextRenderer
from boids import Flock


class Application:
    """Main application managing the game loop and rendering."""
    
    def __init__(self):
        pygame.init()
        pygame.display.set_mode(
            (config.WINDOW["width"], config.WINDOW["height"]),
            DOUBLEBUF | OPENGL
        )
        pygame.display.set_caption(config.WINDOW["title"])
        
        # Core components
        self.camera = Camera()
        self.input_handler = InputHandler(self.camera)
        
        # Rendering components
        self.grid = Grid()
        self.text_renderer = TextRenderer()
        
        # Simulation
        self.flock = Flock(num_boids=config.BOIDS["count"])
        
        # State
        self.clock = pygame.time.Clock()
        self.running = True
        self.fps = 0
        
        self._setup_gl()
    
    def _setup_gl(self):
        """Initialize OpenGL settings."""
        glClearColor(*config.COLORS["background"])
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_FOG)
        glFogi(GL_FOG_MODE, GL_LINEAR)
        glFogfv(GL_FOG_COLOR, config.COLORS["background"])
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(
            config.CAMERA["fov"],
            config.WINDOW["width"] / config.WINDOW["height"],
            config.CAMERA["near_clip"],
            config.CAMERA["far_clip"]
        )
        glMatrixMode(GL_MODELVIEW)
    
    def _update_fog(self):
        """Update fog settings based on camera."""
        glFogf(GL_FOG_START, 50.0)
        glFogf(GL_FOG_END, config.CAMERA["far_clip"] * 0.8)
    
    def _handle_events(self):
        """Process all pending pygame events."""
        for event in pygame.event.get():
            if not self.input_handler.handle_event(event):
                self.running = False
    
    def _update(self, dt: float):
        """Update game state."""
        # Cap dt to prevent physics explosion on lag
        dt = min(dt, 0.05)
        
        self.input_handler.handle_continuous_input(dt)
        self.camera.update(dt)
        self.flock.update(dt)
    
    def _render(self):
        """Render the scene."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.camera.apply()
        self._update_fog()
        
        # Draw world objects
        self.grid.draw(self.camera)
        
        # Pass camera info for rectangular frustum culling
        cam_pos = self.camera.get_position()
        cam_forward, cam_right, cam_up = self.camera.get_camera_axes()
        aspect = config.WINDOW["width"] / config.WINDOW["height"]
        self.flock.draw(cam_pos, cam_forward, cam_right, cam_up, config.CAMERA["fov"], aspect)
        
        # Draw HUD
        screen_size = (config.WINDOW["width"], config.WINDOW["height"])
        visible = self.flock._visible_count
        self.text_renderer.draw_text(
            f"Boids: {visible}/{self.flock.num_boids}  |  FPS: {self.fps:.0f}",
            10, 10, screen_size
        )
        self.text_renderer.draw_text(
            f"θ: {self.camera.theta:.1f}°  φ: {self.camera.phi:.1f}°  Zoom: {self.camera.radius:.1f}",
            10, 35, screen_size
        )
        
        pygame.display.flip()
    
    def run(self):
        """Main application loop."""
        while self.running:
            dt = self.clock.tick() / 1000.0  # Uncapped FPS
            self.fps = self.clock.get_fps()
            
            self._handle_events()
            self._update(dt)
            self._render()
        
        pygame.quit()
