"""Input handling for keyboard and mouse events."""

import pygame
from pygame.locals import *
from config import boids as config

from .camera import Camera


class InputHandler:
    """Handles keyboard and mouse input for camera control."""
    
    def __init__(self, camera: Camera):
        self.camera = camera
        self.mouse_dragging = False
        self.last_mouse_pos = (0, 0)
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Handle a single pygame event.
        Returns False if the application should quit, True otherwise.
        """
        if event.type == QUIT:
            return False
        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                return False
        elif event.type == MOUSEBUTTONDOWN:
            if event.button == 1:
                self.mouse_dragging = True
                self.last_mouse_pos = pygame.mouse.get_pos()
        elif event.type == MOUSEBUTTONUP:
            if event.button == 1:
                self.mouse_dragging = False
        elif event.type == MOUSEWHEEL:
            self.camera.zoom_smooth(-event.y * config.CAMERA["keyboard_zoom_speed"] * 0.5)
        
        return True
    
    def handle_continuous_input(self, dt: float):
        """Handle continuous keyboard input (called each frame)."""
        keys = pygame.key.get_pressed()
        rot_speed = config.CAMERA["keyboard_rotate_speed"] * dt
        zoom_speed = config.CAMERA["keyboard_zoom_speed"] * dt
        
        # Keyboard rotation
        if keys[K_a]:
            self.camera.rotate(-rot_speed, 0)
        if keys[K_d]:
            self.camera.rotate(rot_speed, 0)
        if keys[K_w]:
            self.camera.rotate(0, rot_speed)
        if keys[K_s]:
            self.camera.rotate(0, -rot_speed)
        
        # Keyboard zoom
        if keys[K_q]:
            self.camera.zoom(-zoom_speed)
        if keys[K_e]:
            self.camera.zoom(zoom_speed)
        
        # Mouse drag rotation
        if self.mouse_dragging:
            current_pos = pygame.mouse.get_pos()
            dx = current_pos[0] - self.last_mouse_pos[0]
            dy = current_pos[1] - self.last_mouse_pos[1]
            self.camera.rotate(
                dx * config.CAMERA["mouse_sensitivity"],
                -dy * config.CAMERA["mouse_sensitivity"]
            )
            self.last_mouse_pos = current_pos

