#!/usr/bin/env python3
"""
Convenience entry point for N-body recording.

Usage:
    python record.py                      # Start new recording
    python record.py --resume             # Resume interrupted recording  
    python record.py --status             # Check recording status
    python record.py --list               # List all recordings
    python record.py --resume galaxy_1m   # Resume specific session
"""

from tools.record import main

if __name__ == "__main__":
    main()

