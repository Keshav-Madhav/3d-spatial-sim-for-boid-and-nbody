# 3D N-Body Simulation

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Record Simulation
```bash
python -m tools.record
python -m tools.record --preset-id 6
python -m tools.record --preset quick_galaxy
python -m tools.record --resume
python -m tools.record --list
```

## Playback Recording
```bash
python -m tools.playback <session_name>
python -m tools.playback <session_name> --fps 60
python -m tools.playback <session_name> --loop
```

## Export to Video
```bash
python -m tools.export --list
python -m tools.export <session_name>
python -m tools.export <session_name> --quality high --codec h265
python -m tools.export <session_name> --resolution 4k --fps 60
python -m tools.export <session_name> --camera cinematic
```

## Live Simulation
```bash
python run_nbody.py
python run_nbody.py --bodies 100000 --theta 0.8
```

