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
