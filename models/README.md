# Model weights (not committed to git)

These files are downloaded locally and are git-ignored (see root
`.gitignore`) — they're large binaries with their own license, not source
code.

## `weights/face_landmarker.task`

MediaPipe's face landmark + head-pose model bundle (Tasks API,
`mediapipe.tasks.vision.FaceLandmarker`). Used by `face_detect.py`.

Download it with:

```powershell
python scripts/download_models.py
```

Or manually, from the official MediaPipe model zoo:
`https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`
→ save as `models/weights/face_landmarker.task`.

Run this once per machine (desktop and laptop both need their own local
copy — it is not pushed to GitHub).

## `weights/inswapper_128.onnx`

InsightFace's face-swap model (`insightface.model_zoo.get_model`). Used by
`face_swap.py`. Downloaded the same way as above, via
`python scripts/download_models.py` (community mirror — the official
InsightFace release no longer hosts this file for direct download).

## InsightFace's own `buffalo_l` analysis model

Separately, `face_swap.py` also uses InsightFace's `FaceAnalysis` with the
`buffalo_l` model for face detection/embedding. **This one is not downloaded
by our script** — InsightFace downloads it itself, automatically, to
`~/.insightface/models/buffalo_l` (~290 MB), the first time the app handles
a `/swap` request on a given machine. Expect the first swap request on each
machine to be slower and to need internet access; after that it's cached
locally and instant.

## Lip sync (Wav2Lip) — a separate, git-ignored vendor directory

Unlike the other models, Wav2Lip's source code is not pip-installable and
its GitHub repo has no declared license, so it is never committed here —
it's cloned locally into `vendor/wav2lip/` (git-ignored) with its own
virtual environment, kept separate from this project's own `.venv` to
avoid dependency collisions.

Run once per machine:

```powershell
python scripts/setup_lipsync.py
```

This clones `Rudrabha/Wav2Lip` at a pinned commit, applies two small
compatibility patches (a `librosa` keyword-argument change and a removed
`numpy` type alias), creates `vendor/wav2lip/.venv/`, installs its
dependencies there (including a CUDA build of `torch`), and downloads
`wav2lip_gan.pth` (~415 MB) from a HuggingFace mirror. Also requires
`ffmpeg` on `PATH` (see root `README.md`).
