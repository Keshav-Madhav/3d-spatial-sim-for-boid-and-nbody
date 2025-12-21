"""Grid rendering for spatial reference."""

from OpenGL.GL import *
from config import boids as config


class Grid:
    """Draws a 3D wireframe cube as a spatial reference grid."""
    
    def __init__(self):
        self.base_size = config.GRID["base_size"]
        self.color = config.GRID["color"]
    
    def draw(self, camera=None):
        """
        Draw the grid cube.
        
        Args:
            camera: Optional camera reference (for future LOD or culling)
        """
        e = self.base_size
        
        glBegin(GL_LINES)
        glColor3f(*self.color)
        
        # X-axis edges (front-back lines)
        glVertex3f(-e, -e, -e); glVertex3f(e, -e, -e)
        glVertex3f(-e, e, -e); glVertex3f(e, e, -e)
        glVertex3f(-e, -e, e); glVertex3f(e, -e, e)
        glVertex3f(-e, e, e); glVertex3f(e, e, e)
        
        # Y-axis edges (vertical lines)
        glVertex3f(-e, -e, -e); glVertex3f(-e, e, -e)
        glVertex3f(e, -e, -e); glVertex3f(e, e, -e)
        glVertex3f(-e, -e, e); glVertex3f(-e, e, e)
        glVertex3f(e, -e, e); glVertex3f(e, e, e)
        
        # Z-axis edges (left-right lines)
        glVertex3f(-e, -e, -e); glVertex3f(-e, -e, e)
        glVertex3f(e, -e, -e); glVertex3f(e, -e, e)
        glVertex3f(-e, e, -e); glVertex3f(-e, e, e)
        glVertex3f(e, e, -e); glVertex3f(e, e, e)
        
        glEnd()

