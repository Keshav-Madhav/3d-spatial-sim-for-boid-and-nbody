"""Grid rendering for N-body simulation spatial reference."""

from OpenGL.GL import *
from config import nbody as config


class NBodyGrid:
    """Draws a 3D wireframe cube as a spatial reference grid."""
    
    def __init__(self):
        self.base_size = config.GRID["base_size"]
        self.color = config.GRID["color"]
    
    def draw(self, camera=None):
        """Draw the grid cube."""
        e = self.base_size
        
        glBegin(GL_LINES)
        glColor3f(*self.color)
        
        # X-axis edges
        glVertex3f(-e, -e, -e); glVertex3f(e, -e, -e)
        glVertex3f(-e, e, -e); glVertex3f(e, e, -e)
        glVertex3f(-e, -e, e); glVertex3f(e, -e, e)
        glVertex3f(-e, e, e); glVertex3f(e, e, e)
        
        # Y-axis edges
        glVertex3f(-e, -e, -e); glVertex3f(-e, e, -e)
        glVertex3f(e, -e, -e); glVertex3f(e, e, -e)
        glVertex3f(-e, -e, e); glVertex3f(-e, e, e)
        glVertex3f(e, -e, e); glVertex3f(e, e, e)
        
        # Z-axis edges
        glVertex3f(-e, -e, -e); glVertex3f(-e, -e, e)
        glVertex3f(e, -e, -e); glVertex3f(e, -e, e)
        glVertex3f(-e, e, -e); glVertex3f(-e, e, e)
        glVertex3f(e, e, -e); glVertex3f(e, e, e)
        
        glEnd()

