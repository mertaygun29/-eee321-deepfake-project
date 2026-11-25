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

## Lip sync setup (one extra step, once per machine)

The lip-sync tab needs `ffmpeg` on `PATH` and a separately vendored copy of
Wav2Lip (see `models/README.md` for why it's kept out of this repo).

1. Install ffmpeg:

   ```powershell
   winget install -e --id Gyan.FFmpeg
   ```

   **Restart your terminal after this** — PATH changes from `winget` don't
   apply to an already-open shell. Verify with `ffmpeg -version` in the new
   terminal before continuing.

2. Run the setup script (safe to re-run; downloads ~415 MB + a ~2.9 GB CUDA
   build of `torch` into its own venv, so it takes a while the first time):

   ```powershell
   python scripts/setup_lipsync.py
   ```

   If the machine's GPU/driver doesn't support the CUDA build (`torch.cuda.is_available()`
   prints `False` after setup), fall back to CPU-only — slower, but
   Wav2Lip's own code already auto-detects and uses it:

   ```powershell
   vendor\wav2lip\.venv\Scripts\python.exe -m pip install torch==2.10.0
   ```

## Running tests

```powershell
pytest -v
```

Tests that need a real face photo, or a talking-face video + audio clip,
are skipped unless you add them locally (never commit them — see
`tests/fixtures/README.md`).
