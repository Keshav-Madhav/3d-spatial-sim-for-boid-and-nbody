"""Camera system for 3D navigation."""

import math
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from config import boids as config


class Camera:
    """Orbital camera with smooth zoom and mouse/keyboard controls."""
    
    def __init__(self):
        self.radius = config.CAMERA["initial_radius"]
        self.target_radius = self.radius
        self.theta = config.CAMERA["initial_theta"]
        self.phi = config.CAMERA["initial_phi"]
        self.target = np.array([0.0, 0.0, 0.0])
        self.zoom_smoothing = 8.0
    
    def get_direction(self) -> np.ndarray:
        """Get the normalized direction vector from target to camera."""
        theta_rad = math.radians(self.theta)
        phi_rad = math.radians(self.phi)
        x = math.cos(phi_rad) * math.cos(theta_rad)
        y = math.sin(phi_rad)
        z = math.cos(phi_rad) * math.sin(theta_rad)
        return np.array([x, y, z])
    
    def get_camera_axes(self) -> tuple:
        """
        Get the camera's local coordinate axes (forward, right, up).
        Forward points from camera toward target.
        """
        direction = self.get_direction()
        forward = -direction  # Camera looks opposite to direction vector
        
        # World up
        world_up = np.array([0.0, 1.0, 0.0])
        
        # Right = forward x world_up
        right = np.cross(forward, world_up)
        right_len = np.linalg.norm(right)
        if right_len < 0.001:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right = right / right_len
        
        # Up = right x forward
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        return forward, right, up
    
    def get_position(self) -> np.ndarray:
        """Get the camera's world position."""
        return self.radius * self.get_direction()
    
    def rotate(self, d_theta: float, d_phi: float):
        """Rotate the camera by the given angles in degrees."""
        self.theta = (self.theta + d_theta) % 360
        self.phi = max(
            config.CAMERA["min_phi"], 
            min(config.CAMERA["max_phi"], self.phi + d_phi)
        )
    
    def zoom(self, delta: float):
        """Immediately zoom by the given amount."""
        self.radius = max(
            config.CAMERA["min_radius"], 
            min(config.CAMERA["max_radius"], self.radius + delta)
        )
        self.target_radius = self.radius
    
    def zoom_smooth(self, delta: float):
        """Smoothly zoom by the given amount."""
        self.target_radius = max(
            config.CAMERA["min_radius"], 
            min(config.CAMERA["max_radius"], self.target_radius + delta)
        )
    
    def update(self, dt: float):
        """Update camera state (called each frame)."""
        self.radius += (self.target_radius - self.radius) * self.zoom_smoothing * dt
        self.radius = max(
            config.CAMERA["min_radius"], 
            min(config.CAMERA["max_radius"], self.radius)
        )
    
    def apply(self):
        """Apply the camera transformation to the OpenGL modelview matrix."""
        pos = self.get_position()
        direction = self.get_direction()
        glLoadIdentity()
        
        if self.radius >= 0:
            look_at = self.target
        else:
            look_at = pos - direction * 10
            
        gluLookAt(
            pos[0], pos[1], pos[2],
            look_at[0], look_at[1], look_at[2],
            0, 1, 0
        )

