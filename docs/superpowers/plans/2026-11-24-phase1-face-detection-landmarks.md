# Phase 1 — Image Upload, Face Detection & Landmark Overlay — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Flask web app where a user uploads a photo and sees it back with
facial landmarks and head-pose angles overlaid, satisfying FR1–FR3 of the
Topic 4 spec (upload, face detection, landmarks + head pose).

**Architecture:** One Flask process. `face_detect.py` is a pure processing
module (no Flask dependency) wrapping MediaPipe Face Mesh for landmark
detection and OpenCV `solvePnP` for head-pose estimation. `app.py` exposes
`GET /` (upload page) and `POST /detect` (accepts an image, returns JSON with
landmark count, head-pose angles, and a base64 PNG of the annotated image).
A single HTML page with vanilla JS calls `/detect` and renders the result.

**Tech Stack:** Python, Flask, MediaPipe (`face_mesh`), OpenCV
(`opencv-python`), NumPy, pytest.

## Global Constraints

(From `docs/superpowers/specs/2026-11-24-deepfake-interaction-system-design.md`)

- No model training or training-data collection — only pretrained libraries
  (MediaPipe) — spec §2.
- Uploaded/test face photos must never be committed to the repo — already
  enforced by the root `.gitignore` (blocks `*.jpg`, `*.jpeg`, `*.png` except
  under `docs/`). Test fixture photos go in `tests/fixtures/` and stay local.
- System must eventually run on two machines (desktop with NVIDIA GPU,
  friend's laptop of unknown GPU status) — spec §5. Phase 1 only uses
  CPU-bound MediaPipe, so this constraint doesn't bite yet, but
  `requirements.txt` must not pin anything GPU-only.
- Commit after each task with a message referencing the functional
  requirement(s) it covers — spec §6.
- Remote already configured: `origin` → `https://github.com/mertaygun29/-eee321-deepfake-project.git`, branch `main`.
- Desktop currently has only Python 3.14 installed (via `py` launcher). This
  is newer than most ML libraries officially support — Task 1 verifies this
  works before anything else is built on top of it, with a documented
  fallback.

---

### Task 1: Project skeleton, virtual environment, dependency verification

**Files:**
- Create: `requirements.txt`
- Create: `.venv/` (not committed — already covered by `.gitignore`)
- Create: `scripts/check_env.py`

**Interfaces:**
- Produces: a working `.venv` with Flask, mediapipe, opencv-python, numpy,
  pytest installed, verified importable. All later tasks run inside this
  venv.

- [ ] **Step 1: Create the virtual environment**

Run in the project root (`C:\Users\USER\Documents\eee321-deepfake-project`):

```powershell
py -3.14 -m venv .venv
.venv\Scripts\Activate.ps1
```

- [ ] **Step 2: Write `requirements.txt`**

```
flask==3.1.2
mediapipe==1.2.3
opencv-python==4.12.0.88
numpy==2.2.6
pytest==8.4.2
```

- [ ] **Step 3: Install and check for failures**

```powershell
pip install -r requirements.txt
```

If `mediapipe` or `opencv-python` fails to find a compatible wheel for
Python 3.14 (pip will say something like "no matching distribution"), fall
back to Python 3.12:

```powershell
winget install Python.Python.3.12
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

From here on, every command in this plan assumes the venv (whichever Python
version worked) is activated.

- [ ] **Step 4: Write the environment-check script**

```python
# scripts/check_env.py
import cv2
import flask
import mediapipe as mp
import numpy as np

print("flask", flask.__version__)
print("opencv-python", cv2.__version__)
print("mediapipe", mp.__version__)
print("numpy", np.__version__)

with mp.solutions.face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
    print("FaceMesh initialized OK")
```

- [ ] **Step 5: Run it**

```powershell
python scripts/check_env.py
```

Expected: four version lines followed by `FaceMesh initialized OK`, no
exceptions.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt scripts/check_env.py
git commit -m "Phase 1 setup: venv, dependencies, environment check script"
git push
```

---

### Task 2: Face detection & landmark module (`face_detect.py`)

**Files:**
- Create: `face_detect.py`
- Create: `tests/__init__.py` (empty)
- Create: `tests/fixtures/README.md`
- Test: `tests/test_face_detect.py`

**Interfaces:**
- Produces: `detect_landmarks(image_bgr: np.ndarray) -> tuple[list[tuple[float, float]], np.ndarray]`
  raising `NoFaceDetectedError` (defined in this module) when no face is
  found. Task 3 (`app.py`) imports both `detect_landmarks` and
  `NoFaceDetectedError` from `face_detect`.

- [ ] **Step 1: Document the test fixture requirement**

```markdown
<!-- tests/fixtures/README.md -->
# Test fixtures

Put a real photo with one clearly visible face at `sample_face.jpg` in this
folder to run the tests that need a real face. This file is intentionally
git-ignored (root `.gitignore` blocks `*.jpg`) — never commit real photos.
Tests that need it will skip with a clear message if it's missing.
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_face_detect.py
import pathlib

import numpy as np
import pytest

from face_detect import NoFaceDetectedError, detect_landmarks

FIXTURE_PATH = pathlib.Path(__file__).parent / "fixtures" / "sample_face.jpg"
REQUIRES_FIXTURE = pytest.mark.skipif(
    not FIXTURE_PATH.exists(),
    reason=(
        "Add a real photo with a visible face at "
        "tests/fixtures/sample_face.jpg to run this test "
        "(file is gitignored, add it locally)."
    ),
)


def test_detect_landmarks_raises_on_blank_image():
    blank = np.zeros((200, 200, 3), dtype="uint8")
    with pytest.raises(NoFaceDetectedError):
        detect_landmarks(blank)


@REQUIRES_FIXTURE
def test_detect_landmarks_finds_face():
    import cv2

    image = cv2.imread(str(FIXTURE_PATH))
    landmarks, annotated = detect_landmarks(image)
    assert len(landmarks) == 478  # 468 mesh points + 10 iris points
    assert annotated.shape == image.shape
```

- [ ] **Step 3: Run tests to verify they fail**

```powershell
pytest tests/test_face_detect.py -v
```

Expected: `ModuleNotFoundError: No module named 'face_detect'` (module
doesn't exist yet).

- [ ] **Step 4: Implement `face_detect.py`**

```python
# face_detect.py
import cv2
import mediapipe as mp
import numpy as np

_mp_face_mesh = mp.solutions.face_mesh
_mp_drawing = mp.solutions.drawing_utils
_mp_drawing_styles = mp.solutions.drawing_styles


class NoFaceDetectedError(Exception):
    """Raised when no face can be found in the given image."""


def detect_landmarks(
    image_bgr: np.ndarray,
) -> tuple[list[tuple[float, float]], np.ndarray]:
    """Detect facial landmarks in a BGR image.

    Returns (landmarks, annotated_image_bgr): landmarks is a list of
    (x, y) pixel coordinates for every detected face-mesh point, and
    annotated_image_bgr is a copy of the input with landmark contours
    drawn on it. Raises NoFaceDetectedError if no face is found.
    """
    height, width = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    with _mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        raise NoFaceDetectedError("No face detected in the uploaded image.")

    face_landmarks = results.multi_face_landmarks[0]
    landmarks = [(lm.x * width, lm.y * height) for lm in face_landmarks.landmark]

    annotated = image_bgr.copy()
    _mp_drawing.draw_landmarks(
        image=annotated,
        landmark_list=face_landmarks,
        connections=_mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=_mp_drawing_styles.get_default_face_mesh_contours_style(),
    )
    return landmarks, annotated
```

- [ ] **Step 5: Run tests to verify they pass**

```powershell
pytest tests/test_face_detect.py -v
```

Expected: `test_detect_landmarks_raises_on_blank_image` PASSES.
`test_detect_landmarks_finds_face` either PASSES (if you added a fixture
photo) or SKIPPED (if you didn't) — neither is a failure.

- [ ] **Step 6: Commit**

```bash
git add face_detect.py tests/__init__.py tests/fixtures/README.md tests/test_face_detect.py
git commit -m "FR2-3: add face detection + landmark extraction module"
git push
```

---

### Task 3: Flask backend (`app.py`)

**Files:**
- Create: `app.py`
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: `detect_landmarks`, `NoFaceDetectedError` from `face_detect.py`
  (Task 2).
- Produces: Flask app object `app` (importable as `from app import app`,
  used by `tests/test_app.py` and by `flask run`). Route `POST /detect`
  returns JSON `{"landmark_count": int, "annotated_image_png_base64": str}`
  on success (200), or `{"error": str}` on failure (400 for bad/missing
  upload, 422 for no face found). Task 4 (frontend) relies on exactly these
  field names.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_app.py
import pathlib

import pytest

from app import app as flask_app

FIXTURE_PATH = pathlib.Path(__file__).parent / "fixtures" / "sample_face.jpg"
REQUIRES_FIXTURE = pytest.mark.skipif(
    not FIXTURE_PATH.exists(),
    reason=(
        "Add a real photo with a visible face at "
        "tests/fixtures/sample_face.jpg to run this test."
    ),
)


@pytest.fixture
def client():
    flask_app.config.update(TESTING=True)
    with flask_app.test_client() as test_client:
        yield test_client


def test_detect_route_rejects_missing_image(client):
    response = client.post("/detect", data={})
    assert response.status_code == 400
    assert "error" in response.get_json()


@REQUIRES_FIXTURE
def test_detect_route_returns_landmarks_for_real_face(client):
    with open(FIXTURE_PATH, "rb") as image_file:
        response = client.post(
            "/detect",
            data={"image": (image_file, "sample_face.jpg")},
            content_type="multipart/form-data",
        )
    assert response.status_code == 200
    body = response.get_json()
    assert body["landmark_count"] == 478
    assert "annotated_image_png_base64" in body
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
pytest tests/test_app.py -v
```

Expected: `ModuleNotFoundError: No module named 'app'`.

- [ ] **Step 3: Implement `app.py`**

```python
# app.py
import base64

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from face_detect import NoFaceDetectedError, detect_landmarks

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/detect")
def detect():
    uploaded_file = request.files.get("image")
    if uploaded_file is None or uploaded_file.filename == "":
        return jsonify(error="No image uploaded."), 400

    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return jsonify(error="Could not decode image. Use PNG or JPEG."), 400

    try:
        landmarks, annotated_bgr = detect_landmarks(image_bgr)
    except NoFaceDetectedError as exc:
        return jsonify(error=str(exc)), 422

    success, encoded = cv2.imencode(".png", annotated_bgr)
    if not success:
        return jsonify(error="Failed to encode result image."), 500

    return jsonify(
        landmark_count=len(landmarks),
        annotated_image_png_base64=base64.b64encode(encoded.tobytes()).decode("ascii"),
    )


if __name__ == "__main__":
    app.run(debug=True)
```

- [ ] **Step 4: Minimal template so `index()` doesn't 500 during tests**

```html
<!-- templates/index.html -->
<!doctype html>
<html lang="tr">
<head><meta charset="utf-8"><title>Yüz Tespiti</title></head>
<body><p>placeholder — replaced in Task 4</p></body>
</html>
```

- [ ] **Step 5: Run tests to verify they pass**

```powershell
pytest tests/test_app.py -v
```

Expected: `test_detect_route_rejects_missing_image` PASSES;
`test_detect_route_returns_landmarks_for_real_face` PASSES or SKIPPED.

- [ ] **Step 6: Commit**

```bash
git add app.py templates/index.html tests/test_app.py
git commit -m "FR1-3: add Flask backend with /detect endpoint"
git push
```

---

### Task 4: Frontend — upload form and landmark overlay display

**Files:**
- Modify: `templates/index.html`
- Create: `static/style.css`
- Create: `static/script.js`

**Interfaces:**
- Consumes: `POST /detect` JSON shape from Task 3
  (`{landmark_count, annotated_image_png_base64}` or `{error}`).

- [ ] **Step 1: Replace the placeholder template**

```html
<!-- templates/index.html -->
<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <title>EEE 321 - Yüz Landmark Tespiti</title>
  <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
  <h1>Yüz Tespiti &amp; Landmark Önizleme</h1>
  <form id="upload-form">
    <input type="file" id="image-input" name="image" accept="image/png, image/jpeg" required>
    <button type="submit">Tespit Et</button>
  </form>
  <p id="status"></p>
  <img id="result-image" alt="" hidden>
  <script src="{{ url_for('static', filename='script.js') }}"></script>
</body>
</html>
```

- [ ] **Step 2: Add `static/script.js`**

```javascript
const form = document.getElementById("upload-form");
const statusEl = document.getElementById("status");
const resultImage = document.getElementById("result-image");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const fileInput = document.getElementById("image-input");
  if (!fileInput.files.length) {
    return;
  }

  const formData = new FormData();
  formData.append("image", fileInput.files[0]);

  statusEl.textContent = "İşleniyor...";
  resultImage.hidden = true;

  const response = await fetch("/detect", { method: "POST", body: formData });
  const body = await response.json();

  if (!response.ok) {
    statusEl.textContent = `Hata: ${body.error}`;
    return;
  }

  statusEl.textContent = `${body.landmark_count} landmark noktası bulundu.`;
  resultImage.src = `data:image/png;base64,${body.annotated_image_png_base64}`;
  resultImage.hidden = false;
});
```

- [ ] **Step 3: Add `static/style.css`**

```css
body {
  font-family: system-ui, sans-serif;
  max-width: 720px;
  margin: 2rem auto;
  padding: 0 1rem;
}

#result-image {
  max-width: 100%;
  margin-top: 1rem;
  border: 1px solid #ccc;
}
```

- [ ] **Step 4: Manual browser test (required — this is a UI change)**

```powershell
python app.py
```

Open `http://127.0.0.1:5000/` in a browser:
1. Upload a real photo with a visible face → confirm the status text shows
   a landmark count and the annotated image with contour lines renders.
2. Upload a photo with no face (e.g. a landscape) → confirm the status text
   shows the "No face detected" error, no broken image/crash.
3. Try submitting with no file selected → the browser's native "required"
   validation should block submission.

- [ ] **Step 5: Commit**

```bash
git add templates/index.html static/script.js static/style.css
git commit -m "FR1: add upload UI and landmark overlay display"
git push
```

---

### Task 5: Head-pose estimation (yaw/pitch/roll)

**Files:**
- Modify: `face_detect.py`
- Modify: `app.py`
- Modify: `static/script.js`
- Test: `tests/test_face_detect.py`

**Interfaces:**
- Produces: `estimate_head_pose(landmarks: list[tuple[float, float]], image_shape: tuple[int, int]) -> tuple[float, float, float]`
  returning `(yaw, pitch, roll)` in degrees, added to `face_detect.py`.
- `/detect` response gains a `head_pose` field:
  `{"yaw": float, "pitch": float, "roll": float}`.

- [ ] **Step 1: Add the failing test to `tests/test_face_detect.py`**

```python
@REQUIRES_FIXTURE
def test_estimate_head_pose_returns_three_angles():
    import cv2

    from face_detect import estimate_head_pose

    image = cv2.imread(str(FIXTURE_PATH))
    landmarks, _ = detect_landmarks(image)
    yaw, pitch, roll = estimate_head_pose(landmarks, image.shape)
    for angle in (yaw, pitch, roll):
        assert isinstance(angle, float)
        assert -180.0 <= angle <= 180.0
```

- [ ] **Step 2: Run to verify it fails**

```powershell
pytest tests/test_face_detect.py -v
```

Expected: `ImportError: cannot import name 'estimate_head_pose'`.

- [ ] **Step 3: Add `estimate_head_pose` to `face_detect.py`**

```python
# add to face_detect.py

_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -330.0, -65.0),     # Chin
        (-225.0, 170.0, -135.0),  # Left eye, left corner
        (225.0, 170.0, -135.0),   # Right eye, right corner
        (-150.0, -150.0, -125.0),  # Left mouth corner
        (150.0, -150.0, -125.0),   # Right mouth corner
    ],
    dtype="double",
)

# Indices into the 478-point FaceMesh landmark list for the points above.
_POSE_LANDMARK_INDICES = (1, 152, 33, 263, 61, 291)


def estimate_head_pose(
    landmarks: list[tuple[float, float]], image_shape: tuple[int, int, int]
) -> tuple[float, float, float]:
    """Estimate (yaw, pitch, roll) in degrees from face-mesh landmarks."""
    height, width = image_shape[:2]
    image_points = np.array(
        [landmarks[index] for index in _POSE_LANDMARK_INDICES], dtype="double"
    )

    focal_length = width
    center = (width / 2, height / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype="double",
    )
    dist_coeffs = np.zeros((4, 1))

    _, rotation_vector, _ = cv2.solvePnP(
        _MODEL_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    pitch, yaw, roll = _rotation_matrix_to_euler_angles(rotation_matrix)
    return float(yaw), float(pitch), float(roll)


def _rotation_matrix_to_euler_angles(
    rotation_matrix: np.ndarray,
) -> tuple[float, float, float]:
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = 0.0

    return float(np.degrees(x)), float(np.degrees(y)), float(np.degrees(z))
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
pytest tests/test_face_detect.py -v
```

Expected: all tests PASS or SKIP (none FAIL).

- [ ] **Step 5: Wire it into `app.py`**

In `app.py`, update the import and the success branch of `detect()`:

```python
from face_detect import NoFaceDetectedError, detect_landmarks, estimate_head_pose
```

```python
    try:
        landmarks, annotated_bgr = detect_landmarks(image_bgr)
    except NoFaceDetectedError as exc:
        return jsonify(error=str(exc)), 422

    yaw, pitch, roll = estimate_head_pose(landmarks, image_bgr.shape)

    success, encoded = cv2.imencode(".png", annotated_bgr)
    if not success:
        return jsonify(error="Failed to encode result image."), 500

    return jsonify(
        landmark_count=len(landmarks),
        annotated_image_png_base64=base64.b64encode(encoded.tobytes()).decode("ascii"),
        head_pose={"yaw": yaw, "pitch": pitch, "roll": roll},
    )
```

- [ ] **Step 6: Display it in the UI — update `static/script.js`**

In the success branch, after setting `statusEl.textContent`, add:

```javascript
  const pose = body.head_pose;
  statusEl.textContent += ` (Yaw: ${pose.yaw.toFixed(1)}°, Pitch: ${pose.pitch.toFixed(1)}°, Roll: ${pose.roll.toFixed(1)}°)`;
```

- [ ] **Step 7: Manual browser re-test**

```powershell
python app.py
```

Upload the same real-photo test case as Task 4 and confirm the status line
now also shows yaw/pitch/roll numbers that change sensibly when you upload a
frontal vs. tilted photo of the same face.

- [ ] **Step 8: Commit**

```bash
git add face_detect.py app.py static/script.js tests/test_face_detect.py
git commit -m "FR3: add head-pose estimation and display"
git push
```

---

### Task 6: README — setup instructions for both machines

**Files:**
- Modify: `README.md`

**Interfaces:** none (documentation only).

- [ ] **Step 1: Replace `README.md` contents**

```markdown
# EEE 321 DSP Project — AI-Based Deepfake Interaction System for Entertainment

Solo course project (Topic 4). Design spec:
`docs/superpowers/specs/2026-11-24-deepfake-interaction-system-design.md`.
Phase 1 plan: `docs/superpowers/plans/2026-11-24-phase1-face-detection-landmarks.md`.

## Current status: Phase 1

Upload a photo, detect facial landmarks (MediaPipe Face Mesh) and head pose
(yaw/pitch/roll via OpenCV `solvePnP`), see the result overlaid in the
browser.

## Setup (desktop or laptop — same steps on both)

```powershell
git clone https://github.com/mertaygun29/-eee321-deepfake-project.git
cd -eee321-deepfake-project
py -3.14 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
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
```

## Running tests

```powershell
pytest -v
```

Tests that need a real face photo are skipped unless you add one locally at
`tests/fixtures/sample_face.jpg` (never commit it — see
`tests/fixtures/README.md`).
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "Document Phase 1 setup and run instructions for both machines"
git push
```
