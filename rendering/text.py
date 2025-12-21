"""Text rendering for HUD elements."""

import pygame
from OpenGL.GL import *


class TextRenderer:
    """Renders text overlays using pygame fonts and OpenGL."""
    
    def __init__(self, font_name: str = "monospace", font_size: int = 18):
        pygame.font.init()
        self.font = pygame.font.SysFont(font_name, font_size)
    
    def draw_text(self, text: str, x: int, y: int, screen_size: tuple):
        """
        Draw text at the given screen position.
        
        Args:
            text: The string to render
            x: X position from left edge
            y: Y position from top edge
            screen_size: (width, height) of the screen
        """
        text_surface = self.font.render(text, True, (230, 230, 230))
        text_data = pygame.image.tostring(text_surface, "RGBA", True)
        w, h = text_surface.get_size()
        
        # Switch to orthographic projection for 2D rendering
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, screen_size[0], 0, screen_size[1], -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Draw the text with alpha blending
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glRasterPos2f(x, screen_size[1] - y - h)
        glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, text_data)
        glDisable(GL_BLEND)
        
        # Restore projection
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

