# AI-Based Deepfake Interaction System for Entertainment — Design Spec

**Date:** 2026-11-24
**Course:** EEE 321 DSP Project (Topic 4)
**Author:** USER (solo project)
**Checkpoint:** Partial demo due Friday (this week); full system due end of semester.

## 1. Context

Course gives 4 topic options; this project targets Topic 4, "AI-Based Deepfake
Interaction System for Entertainment" (full requirements in `proje.pdf`, 15
functional requirements: face detection/tracking, landmark detection, face
swap, reenactment, lip-sync, expression transfer, temporal consistency,
seamless blending, real-time operation, parameter controls, A/V sync,
before/after preview, evaluation metrics, and a mandatory AI-generated
disclosure/watermark).

Instructor explicitly said students do not need to implement DSP/ML models
from scratch — using existing applications/AI tools is acceptable. Given the
timeline (solo student, Friday checkpoint, end-of-semester full delivery),
this spec scopes the 15 requirements into phases and picks concrete
off-the-shelf tools to wire together rather than training anything from
scratch.

## 2. Constraints

- Solo project, developed primarily on a desktop PC with an NVIDIA GPU.
- Must also run on a friend's laptop for the Friday demo — laptop GPU
  presence is unknown, so the system must degrade gracefully to CPU.
- Deliverable must show visible progress via git commit history, viewable by
  the instructor without a GitHub account → **public** GitHub repo.
- No training data collection or model training — only pretrained, openly
  available models.
- Uploaded face images/outputs (real people) must never be committed to the
  public repo — privacy/consent concern, separate from code.
- Every generated/exported result must carry a visible "AI GENERATED"
  disclosure overlay (functional requirement 15) — non-negotiable from the
  first version that produces output, not deferred to the end.

## 3. Architecture

Single Flask (Python) web application:

- **Frontend:** one HTML page + vanilla JS/CSS. Upload form(s), a canvas/img
  preview area, buttons to trigger each backend operation, before/after
  side-by-side view.
- **Backend (Flask, `app.py`):** HTTP routes that accept uploaded images,
  call processing modules, and return result images/JSON to the frontend.
- **Processing modules:**
  - `face_detect.py` — face detection + landmark extraction via MediaPipe
    Face Mesh. Returns landmark coordinates and a landmark-annotated image.
  - `face_swap.py` — face swap via InsightFace (`buffalo_l` analysis model
    for detection/embedding + `inswapper_128` swap model), executed through
    onnxruntime.
  - `watermark.py` — overlays a visible "AI GENERATED" text/logo on any
    exported/result image before it's returned or saved.
- **Model weights:** `inswapper_128.onnx` and MediaPipe/InsightFace
  analysis weights are downloaded locally on first run, never committed.
  `models/README.md` documents the download source and expected file paths.
- **Execution provider selection:** onnxruntime picks
  `CUDAExecutionProvider` if available, otherwise falls back to
  `CPUExecutionProvider` automatically — same code path on both machines.

## 4. Scope by phase

Mapped to the 15 functional requirements in `proje.pdf`; not all 15 are
targeted for a solo-project timeline — Phase 4 is explicitly a stretch/likely
cut, to be reported honestly as future work if not reached.

- **Phase 1 — Foundation (due Friday).** FR1–FR3: image upload (PNG/JPEG),
  face detection, facial landmark extraction, landmarks displayed as an
  overlay in the browser. No swap yet.
- **Phase 2 — Core deepfake feature.** FR4: face swap between an uploaded
  source face and a target image. FR15: every swap result carries the
  disclosure watermark from day one.
- **Phase 3 — Comparison & evaluation.** FR13: before/after side-by-side
  view. FR14 (partial): basic quality/speed metrics shown in the UI
  (processing time; optionally a similarity/quality score).
- **Phase 4 — Stretch, likely out of scope.** FR5–FR12 (reenactment,
  lip-sync, expression transfer, real-time video, temporal consistency across
  frames). Only attempted if Phases 1–3 are solid and time remains; otherwise
  documented as future work in the final report.

## 5. Portability (two machines)

- GitHub is the transport mechanism between the desktop and the friend's
  laptop — no USB/manual copying.
- `requirements.txt` pins Python dependencies (Flask, mediapipe,
  opencv-python, insightface, onnxruntime or onnxruntime-gpu).
- `README.md` documents identical setup steps for both machines: clone →
  create venv → `pip install -r requirements.txt` → download model weights
  per `models/README.md` → run.
- Before Friday: do one full dry-run clone+setup on the laptop (or an
  equivalent clean environment) ahead of time, not for the first time during
  the demo.
- Known risk: desktop currently has Python 3.14, which is very new and may
  not yet have prebuilt wheels for mediapipe/onnxruntime/insightface. This
  must be checked during environment setup (next step, not this spec) and a
  compatible Python version (e.g. 3.11/3.12 via a dedicated venv) used if
  needed.

## 6. Git workflow

- `git init` in the project root; public GitHub repo, link shared with the
  instructor so commit history is visible without requiring them to have a
  GitHub account.
- `.gitignore` excludes: virtual environment, `__pycache__`, model weight
  files, and any user-uploaded or generated images (privacy/consent — real
  faces never land in a public repo).
- Commit after each meaningful implementation step, with messages that
  reference the functional requirement(s) covered (e.g.
  `"FR1-3: image upload + face detection + landmark overlay"`), so the
  instructor's commit-history review doubles as a progress log.

## 7. Error handling & limitations

- No face detected in an uploaded image → user-facing error message in the
  UI, not a server crash.
- GPU unavailable → silent fallback to CPU (slower, logged, not an error).
- Model weight files missing on first run → clear setup instructions
  surfaced to the user (via `models/README.md`), not a cryptic stack trace.
- Explicitly out of scope for this project: processing images/videos of
  people without their consent. The system is for course demonstration using
  the student's own images/consenting subjects only.

## 8. Testing / demo plan

- Manual verification per phase: upload a test image, confirm landmarks
  render correctly (Phase 1); run a swap between two sample images and
  confirm the watermark appears (Phase 2); confirm before/after view and
  timing display (Phase 3).
- Dry-run full setup on a second machine (ideally the actual laptop) before
  Friday to catch environment issues early.
