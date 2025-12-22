"""
Hot-reload development runner for N-body simulation.

Watches source files and restarts the application when changes are detected.
Uses a debounce timer to avoid rapid restarts during active editing.

Controls:
    - Press Enter or 'r': Force immediate reload
    - Ctrl+C: Exit
"""

import subprocess
import sys
import time
import os
from pathlib import Path
from threading import Thread
from queue import Queue, Empty

# Platform-specific imports
if sys.platform != 'win32':
    import select
    import termios
    import tty

# Configuration
DEBOUNCE_SECONDS = 3.0  # Shorter debounce for faster iteration
POLL_INTERVAL = 0.2

# Files and directories to watch for changes
WATCH_PATTERNS = [
    "nbody_main.py",
    "config/*.py",
    "core/*.py",
    "rendering/*.py",
    "nbody/*.py",
    "tools/*.py",
]


def get_watched_files():
    """Get all files matching watch patterns."""
    root = Path(__file__).parent
    files = set()
    
    for pattern in WATCH_PATTERNS:
        files.update(root.glob(pattern))
    
    return files


def get_mtimes():
    """Get modification times for all watched files."""
    mtimes = {}
    for f in get_watched_files():
        if f.exists():
            mtimes[str(f)] = f.stat().st_mtime
    return mtimes


def get_changed_files(old_mtimes, new_mtimes):
    """Find which files changed between two mtime snapshots."""
    changed = set(new_mtimes.keys()) ^ set(old_mtimes.keys())
    for f in new_mtimes:
        if f in old_mtimes and new_mtimes[f] != old_mtimes[f]:
            changed.add(f)
    return changed


def input_listener(queue):
    """Background thread to listen for keyboard input."""
    if sys.platform == 'win32':
        # Windows implementation using msvcrt
        import msvcrt
        try:
            while True:
                if msvcrt.kbhit():
                    char = msvcrt.getch().decode('utf-8', errors='ignore')
                    queue.put(char)
                    if char == '\x03':  # Ctrl+C
                        break
                time.sleep(0.1)
        except Exception:
            pass
    else:
        # Unix implementation using termios
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while True:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    char = sys.stdin.read(1)
                    queue.put(char)
                    if char == '\x03':  # Ctrl+C
                        break
        except Exception:
            pass
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def start_process():
    """Start the N-body application process."""
    return subprocess.Popen([sys.executable, "nbody_main.py"])


def stop_process(process):
    """Stop the running process."""
    if process:
        process.terminate()
        process.wait()


def main():
    process = None
    last_mtimes = {}
    pending_reload = False
    reload_at = 0
    changed_files = set()
    
    # Start input listener thread
    input_queue = Queue()
    input_thread = Thread(target=input_listener, args=(input_queue,), daemon=True)
    input_thread.start()
    
    print("[run] N-body hot-reload runner started")
    print(f"[run] Debounce: {DEBOUNCE_SECONDS}s | Press Enter/r to force reload | Ctrl+C to exit\n")
    
    # Initial start
    print("[run] Starting N-body simulation...")
    process = start_process()
    last_mtimes = get_mtimes()
    
    while True:
        current_time = time.time()
        
        # Check for keyboard input
        try:
            while True:
                char = input_queue.get_nowait()
                if char in ('\n', '\r', 'r', 'R'):
                    print("\n[reload] Manual reload triggered")
                    stop_process(process)
                    process = start_process()
                    last_mtimes = get_mtimes()
                    pending_reload = False
                    changed_files.clear()
                elif char == '\x03':  # Ctrl+C
                    raise KeyboardInterrupt
        except Empty:
            pass
        
        # Check for file changes
        current_mtimes = get_mtimes()
        new_changes = get_changed_files(last_mtimes, current_mtimes)
        
        if new_changes:
            changed_files.update(new_changes)
            last_mtimes = current_mtimes
            reload_at = current_time + DEBOUNCE_SECONDS
            
            if not pending_reload:
                pending_reload = True
                changed_names = [Path(f).name for f in changed_files]
                print(f"\n[watch] Changed: {', '.join(changed_names)}")
                print(f"[watch] Reloading in {DEBOUNCE_SECONDS:.0f}s... (press Enter to reload now)")
            else:
                # Additional changes during debounce
                changed_names = [Path(f).name for f in new_changes]
                remaining = reload_at - current_time
                print(f"[watch] +{', '.join(changed_names)} (reload in {remaining:.0f}s)")
        
        # Execute pending reload after debounce
        if pending_reload and current_time >= reload_at:
            changed_names = [Path(f).name for f in changed_files]
            print(f"\n[reload] Restarting: {', '.join(changed_names)}")
            stop_process(process)
            process = start_process()
            pending_reload = False
            changed_files.clear()
        
        try:
            time.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            stop_process(process)
            print("\n[exit]")
            break
        
        # Exit if the process ended on its own
        if process and process.poll() is not None:
            print("[exit] Application closed")
            break


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()

