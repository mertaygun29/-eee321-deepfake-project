# EEE 321 DSP Project — AI-Based Deepfake Interaction System for Entertainment

Solo course project (Topic 4). Design spec:
`docs/superpowers/specs/2026-11-24-deepfake-interaction-system-design.md`.
Phase 1 plan: `docs/superpowers/plans/2026-11-24-phase1-face-detection-landmarks.md`.
Phase 2 plan: `docs/superpowers/plans/2026-11-24-phase2-face-swap.md`.
Phase 3 plan: `docs/superpowers/plans/2026-11-25-phase3-comparison-and-metrics.md`.
Phase 4a plan: `docs/superpowers/plans/2026-11-25-phase4a-lip-sync.md`.

## Current status: Phase 4a

- **Landmark detection:** upload a photo, detect facial landmarks (MediaPipe
  FaceLandmarker) and head pose (yaw/pitch/roll via OpenCV `solvePnP`), see
  a before/after view in the browser with processing time.
- **Face swap:** upload a source face and a target photo, get back the
  target with the source's face swapped in (InsightFace `inswapper_128`),
  always with a visible "AI GENERATED" disclosure watermark, shown
  before/after with processing time and an identity-similarity score
  (cosine similarity between the source face and the swapped result).
- **Lip sync:** upload a video (a talking face) and a separate audio file,
  get back the video with its mouth re-synced to that audio (Wav2Lip),
  always with the same visible "AI GENERATED" disclosure, shown
  before/after with processing time. Needs one extra setup step — see
  below.

## Prerequisites (do this first, on both machines)

Install all three, in one go if none are present yet:

```powershell
winget install -e --id Git.Git
winget install -e --id Python.Python.3.14
winget install -e --id Gyan.FFmpeg
```

(Python 3.14 is confirmed working for every dependency in this project,
including the separate lip-sync venv below — the Python 3.12 fallback a few
paragraphs down is a defensive option, not something hit during
development.)

**Then close and reopen your terminal** — `winget` updates PATH, but an
already-open terminal doesn't see that change. Verify in the *new* terminal:

```powershell
git --version
py --version
ffmpeg -version
```

All three should print a version, not "not recognized". Only move on once
they do.

## Setup (desktop or laptop — same steps on both)

```powershell
git clone https://github.com/mertaygun29/-eee321-deepfake-project.git
cd -eee321-deepfake-project
py -3.14 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/download_models.py
```

If `pip install` fails to find a wheel for `mediapipe` or `opencv-python` on
your Python version, install Python 3.12 instead and recreate the venv:

```powershell
winget install -e --id Python.Python.3.12
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

At this point `python app.py` already works for the detect and swap tabs —
the lip-sync tab needs the extra step below first, or it'll just show a
clear "not set up yet" message instead of a result.

## Lip sync setup (one extra step, once per machine)

The lip-sync tab needs a separately vendored copy of Wav2Lip (see
`models/README.md` for why it's kept out of this repo) — `ffmpeg` is already
covered by Prerequisites above.

Run the setup script (safe to re-run; downloads ~415 MB + a ~2.9 GB CUDA
build of `torch` into its own venv, so it takes a while the first time):

```powershell
python scripts/setup_lipsync.py
```

Then check whether the machine's GPU/driver actually supports the CUDA
build that was just installed:

```powershell
vendor\wav2lip\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

- Prints `True` → GPU is working, nothing else to do.
- Prints `False` → run this fallback command once. It replaces the CUDA
  build with a CPU-only one; Wav2Lip's own code already auto-detects and
  uses whichever is installed, so no other change is needed — lip sync
  will just run slower:

  ```powershell
  vendor\wav2lip\.venv\Scripts\python.exe -m pip install torch==2.10.0
  ```

## Run it

Every time you open a new terminal (the venv isn't active by default):

```powershell
.venv\Scripts\Activate.ps1
python app.py
```

Wait for a line like `Running on http://127.0.0.1:5000`, then open that
address in a browser. Stop the server with `Ctrl+C` in the terminal.

## Running tests

```powershell
pytest -v
```

Tests that need a real face photo, or a talking-face video + audio clip,
are skipped unless you add them locally (never commit them — see
`tests/fixtures/README.md`).
