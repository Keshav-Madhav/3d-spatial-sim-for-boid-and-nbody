"""Individual boid entity with position, velocity, and behaviors."""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Boid:
    """
    A single boid (bird-oid object) in the simulation.
    
    Attributes:
        position: 3D position vector
        velocity: 3D velocity vector
        acceleration: 3D acceleration vector (reset each frame)
        color: RGB color tuple (0-1 range)
        max_speed: Maximum velocity magnitude
        max_force: Maximum steering force magnitude
    """
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    max_speed: float = 40.0
    max_force: float = 80.0
    
    def apply_force(self, force: np.ndarray):
        """Add a force to the boid's acceleration."""
        self.acceleration += force
    
    def update(self, dt: float):
        """Update position and velocity based on acceleration."""
        self.velocity += self.acceleration * dt
        
        # Limit speed
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = (self.velocity / speed) * self.max_speed
        
        self.position += self.velocity * dt
        
        # Reset acceleration for next frame
        self.acceleration = np.zeros(3)
    
    def seek(self, target: np.ndarray) -> np.ndarray:
        """Calculate steering force toward a target."""
        desired = target - self.position
        dist = np.linalg.norm(desired)
        
        if dist > 0:
            desired = (desired / dist) * self.max_speed
        
        steer = desired - self.velocity
        steer_mag = np.linalg.norm(steer)
        
        if steer_mag > self.max_force:
            steer = (steer / steer_mag) * self.max_force
        
        return steer
    
    def avoid_walls(self, bounds: float, margin: float = 15.0) -> np.ndarray:
        """
        Calculate steering force to avoid walls.
        
        Args:
            bounds: The boundary limit (+/- bounds on each axis)
            margin: Distance from wall to start turning
            
        Returns:
            Steering force vector
        """
        steer = np.zeros(3)
        
        for i in range(3):
            if self.position[i] > bounds - margin:
                # Too close to positive boundary, steer negative
                strength = (self.position[i] - (bounds - margin)) / margin
                steer[i] = -self.max_force * min(strength * 2, 1.0)
            elif self.position[i] < -bounds + margin:
                # Too close to negative boundary, steer positive
                strength = ((-bounds + margin) - self.position[i]) / margin
                steer[i] = self.max_force * min(strength * 2, 1.0)
        
        return steer
