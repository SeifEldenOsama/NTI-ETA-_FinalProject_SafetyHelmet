# SafetyHelmet

This project implements a YOLO-based Safety Helmet Detection system optimized with TFLite. It provides:
- A Python package for detection logic (`src/safetyhelmet`)
- A Gradio web interface to demo detection (`app/gradio_app.py`)
- Scripts to convert/export models to TFLite (`scripts/convert_to_tflite.py`)
- Notebooks for exploration and training in `notebooks/`

Quickstart:
1. Create a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. If you have an existing notebook, convert and extract functions into `src/safetyhelmet`.
   Recommended: use jupytext to keep notebook <-> script synced.
3. Place your trained model(s) in `models/` (e.g. `models/model.tflite`).
4. Run demo:
   ```
   python app/gradio_app.py
   ```
5. Run tests:
   ```
   pytest
   ```

Project layout
- `src/safetyhelmet/` — package code (model loading, inference, utilities)
- `app/` — Gradio demo script
- `scripts/` — conversion and utility scripts
- `models/` — store TFLite/SavedModel files (do not commit large binaries; use Git LFS for large files)
- `notebooks/` — analysis / training notebooks

Notes:
- Use Git LFS for storing large model binaries.
- Add CI to run tests and basic linting (example in `.github/workflows/ci.yml`).