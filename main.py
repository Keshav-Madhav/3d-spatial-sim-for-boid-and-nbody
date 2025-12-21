"""
3D Boids Simulation
===================

A real-time flocking simulation with 3D orbital camera controls.

Controls:
    - W/S: Rotate camera up/down
    - A/D: Rotate camera left/right
    - Q/E: Zoom in/out
    - Mouse drag: Rotate camera
    - Mouse wheel: Zoom
    - ESC: Quit
"""

from core import Application


def main():
    app = Application()
    app.run()


if __name__ == "__main__":
    main()
