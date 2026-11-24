# EEE 321 DSP Project — AI-Based Deepfake Interaction System for Entertainment

Solo course project (Topic 4). Design spec:
`docs/superpowers/specs/2026-11-24-deepfake-interaction-system-design.md`.
Phase 1 plan: `docs/superpowers/plans/2026-11-24-phase1-face-detection-landmarks.md`.
Phase 2 plan: `docs/superpowers/plans/2026-11-24-phase2-face-swap.md`.

## Current status: Phase 2

- **Landmark detection:** upload a photo, detect facial landmarks (MediaPipe
  FaceLandmarker) and head pose (yaw/pitch/roll via OpenCV `solvePnP`), see
  the result overlaid in the browser.
- **Face swap:** upload a source face and a target photo, get back the
  target with the source's face swapped in (InsightFace `inswapper_128`),
  always with a visible "AI GENERATED" disclosure watermark.

## Setup (desktop or laptop — same steps on both)

```powershell
git clone https://github.com/mertaygun29/-eee321-deepfake-project.git
cd -eee321-deepfake-project
py -3.14 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/download_models.py
python app.py
```

Open `http://127.0.0.1:5000/` in a browser.

If `pip install` fails to find a wheel for `mediapipe` or `opencv-python` on
your Python version, install Python 3.12 instead and recreate the venv:

```powershell
winget install Python.Python.3.12
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/download_models.py
```

`scripts/download_models.py` fetches `models/weights/face_landmarker.task`
(~3.7 MB) and `models/weights/inswapper_128.onnx` (~550 MB) — both
git-ignored, run this once per machine, it won't come from `git clone`. The
first `/swap` request on each machine will also trigger a separate, automatic
~290 MB download of InsightFace's own `buffalo_l` model to `~/.insightface`
— see `models/README.md` for details.

## Running tests

```powershell
pytest -v
```

Tests that need a real face photo are skipped unless you add one locally at
`tests/fixtures/sample_face.jpg` (never commit it — see
`tests/fixtures/README.md`).
