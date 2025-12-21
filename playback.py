#!/usr/bin/env python3
"""
Convenience entry point for N-body playback.

Usage:
    python playback.py <session_name>              # Playback recording
    python playback.py <session_name> --fps 60    # Custom FPS
    python playback.py <session_name> --loop      # Loop playback
    python playback.py <session_name> --export    # Export to MP4
"""

from tools.playback import main

if __name__ == "__main__":
    main()

