# Phase 2 — Face Swap & Disclosure Watermark — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a face-swap feature to the existing Flask app — upload a source
face and a target photo, get back the target with the source's face swapped
in, always carrying a visible "AI GENERATED" disclosure watermark. Satisfies
FR4 and FR15 of the Topic 4 spec.

**Architecture:** Two new pure modules alongside the existing `face_detect.py`:
`face_swap.py` (wraps InsightFace's `FaceAnalysis` + `inswapper_128` ONNX
model) and `watermark.py` (burns a visible disclosure label into any result
image). `app.py` gains a `POST /swap` route. The existing upload page gains a
second form for the swap feature; the Phase 1 landmark feature is untouched.

**Tech Stack:** insightface 0.7.3 (`FaceAnalysis`, `model_zoo.get_model`),
onnxruntime 1.32.0 (CPU execution provider), OpenCV, Flask.

## Global Constraints

(From `docs/superpowers/specs/2026-11-24-deepfake-interaction-system-design.md`)

- Every exported/result image from the swap feature must carry a visible
  "AI GENERATED" disclosure watermark — spec §2, non-negotiable, applied in
  this phase's very first working version, not deferred.
- Uploaded/result face photos must never be committed to the repo — enforced
  by the existing root `.gitignore`.
- No model training — only pretrained libraries (InsightFace) — spec §2.
- Commit after each task with a message referencing the functional
  requirement(s) it covers — spec §6.

**Deviation from the original spec, discovered during this plan's research
(documented here per the spec's intent, not silently): the spec assumed we'd
need to auto-detect and prefer a CUDA (GPU) execution provider, falling back
to CPU.** Measured swap inference on this desktop's CPU takes ~0.3 seconds
per image with the `inswapper_128` model — fast enough that GPU adds
unnecessary setup complexity (installing `onnxruntime-gpu` instead of
`onnxruntime` is an extra per-machine choice, and the two packages conflict
if both installed). This plan uses `CPUExecutionProvider` unconditionally on
both machines. If a later phase needs heavier models where this matters,
revisit then.

**Also discovered during research:** `opencv-python==4.12.0.88` (pinned in
Phase 1) caps `numpy<2.3.0`, which conflicts with the `numpy==2.5.5` pin
needed to fix the Python 3.14 longdouble bug from Phase 1. Bumped to
`opencv-python==5.1.0.95`, which only requires `numpy>=2` with no upper
bound. Verified the full Phase 1 test suite and the InsightFace pipeline
both still work against this version.

**Also discovered during research:** InsightFace's own `buffalo_l` face
analysis model is NOT part of our `scripts/download_models.py` — the
`FaceAnalysis` class downloads it automatically (to `~/.insightface/models/`,
outside the project folder) the first time `.prepare()` runs. This is
InsightFace's own caching behavior, not something we need to replicate; it
just means the very first `/swap` request on each machine will be slower
(one-time ~290 MB download) and needs internet access. Documented in
`models/README.md` (Task 1) so this isn't surprising on the laptop.

---

### Task 1: Dependencies and model weights

**Files:**
- Modify: `requirements.txt`
- Modify: `scripts/download_models.py`
- Modify: `models/README.md`

**Interfaces:**
- Produces: `models/weights/inswapper_128.onnx` present on disk after
  running the download script. Task 2 (`face_swap.py`) depends on this file
  existing at that exact path.

- [ ] **Step 1: Add dependencies to `requirements.txt`**

```
flask==3.1.2
mediapipe==1.2.3
opencv-python==5.1.0.95
numpy==2.5.5
pytest==8.4.2
insightface==0.7.3
onnxruntime==1.32.0
```

- [ ] **Step 2: Add the inswapper download entry to `scripts/download_models.py`**

Add this key to the existing `MODELS` dict (keep the `face_landmarker.task`
entry as-is):

```python
    # Official InsightFace releases no longer host this file directly;
    # this is the community mirror used by most face-swap projects (roop,
    # ReActor, etc.) for an unauthenticated direct download.
    "inswapper_128.onnx": (
        "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/"
        "inswapper_128.onnx"
    ),
```

- [ ] **Step 3: Run it**

```powershell
pip install -r requirements.txt
python scripts/download_models.py
```

Expected: `face_landmarker.task already present, skipping.` followed by a
download of `inswapper_128.onnx` (~550 MB — this will take a while on a slow
connection; that's expected, it's a one-time per-machine step).

- [ ] **Step 4: Document both model-loading behaviors in `models/README.md`**

Add this section (keep the existing `face_landmarker.task` section above it):

```markdown
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
```

- [ ] **Step 5: Commit**

```bash
git add requirements.txt scripts/download_models.py models/README.md
git commit -m "FR4 setup: add insightface/onnxruntime deps and inswapper_128 download"
```

---

### Task 2: Face-swap module (`face_swap.py`)

**Files:**
- Create: `face_swap.py`
- Test: `tests/test_face_swap.py`

**Interfaces:**
- Consumes: `NoFaceDetectedError` from `face_detect.py` (reused, so both
  detection and swap failures look the same to `app.py`).
- Produces: `swap_faces(source_bgr: np.ndarray, target_bgr: np.ndarray) -> np.ndarray`,
  returning an image the same shape as `target_bgr`. Task 4 (`app.py`)
  imports this.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_face_swap.py
import pathlib

import cv2
import numpy as np
import pytest

from face_detect import NoFaceDetectedError
from face_swap import swap_faces

FIXTURE_PATH = pathlib.Path(__file__).parent / "fixtures" / "sample_face.jpg"
REQUIRES_FIXTURE = pytest.mark.skipif(
    not FIXTURE_PATH.exists(),
    reason=(
        "Add a real photo with a visible face at "
        "tests/fixtures/sample_face.jpg to run this test."
    ),
)


def test_swap_faces_raises_when_source_has_no_face():
    blank = np.zeros((200, 200, 3), dtype="uint8")
    with pytest.raises(NoFaceDetectedError):
        swap_faces(blank, blank)


@REQUIRES_FIXTURE
def test_swap_faces_returns_image_same_shape_as_target():
    image = cv2.imread(str(FIXTURE_PATH))
    result = swap_faces(image, image)
    assert result.shape == image.shape
    assert result.dtype == image.dtype
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
pytest tests/test_face_swap.py -v
```

Expected: `ModuleNotFoundError: No module named 'face_swap'`.

- [ ] **Step 3: Implement `face_swap.py`**

```python
# face_swap.py
import pathlib

import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

from face_detect import NoFaceDetectedError

_SWAPPER_MODEL_PATH = (
    pathlib.Path(__file__).parent / "models" / "weights" / "inswapper_128.onnx"
)

_analyzer = None
_swapper = None


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        _analyzer = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        _analyzer.prepare(ctx_id=0, det_size=(640, 640))
    return _analyzer


def _get_swapper():
    global _swapper
    if _swapper is None:
        if not _SWAPPER_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Missing model file: {_SWAPPER_MODEL_PATH}. "
                "Run `python scripts/download_models.py` first."
            )
        _swapper = get_model(str(_SWAPPER_MODEL_PATH), providers=["CPUExecutionProvider"])
    return _swapper


def swap_faces(source_bgr: np.ndarray, target_bgr: np.ndarray) -> np.ndarray:
    """Swap the face from source_bgr onto the face found in target_bgr.

    Returns the swapped image (same shape as target_bgr). Raises
    NoFaceDetectedError if either image has no detectable face.
    """
    analyzer = _get_analyzer()

    source_faces = analyzer.get(source_bgr)
    if not source_faces:
        raise NoFaceDetectedError("No face detected in the source image.")

    target_faces = analyzer.get(target_bgr)
    if not target_faces:
        raise NoFaceDetectedError("No face detected in the target image.")

    return _get_swapper().get(
        target_bgr, target_faces[0], source_faces[0], paste_back=True
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
pytest tests/test_face_swap.py -v
```

Expected: `test_swap_faces_raises_when_source_has_no_face` PASSES.
`test_swap_faces_returns_image_same_shape_as_target` PASSES or SKIPPED.

- [ ] **Step 5: Commit**

```bash
git add face_swap.py tests/test_face_swap.py
git commit -m "FR4: add face-swap module wrapping InsightFace inswapper"
```

---

### Task 3: Disclosure watermark module (`watermark.py`)

**Files:**
- Create: `watermark.py`
- Test: `tests/test_watermark.py`

**Interfaces:**
- Produces: `add_disclosure_watermark(image_bgr: np.ndarray) -> np.ndarray`,
  returning a copy of the input with a visible watermark burned in, same
  shape and dtype. Task 4 (`app.py`) imports this.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_watermark.py
import numpy as np

from watermark import add_disclosure_watermark


def test_add_disclosure_watermark_changes_pixels_without_resizing():
    image = np.zeros((400, 400, 3), dtype="uint8")
    watermarked = add_disclosure_watermark(image)
    assert watermarked.shape == image.shape
    assert watermarked.dtype == image.dtype
    assert not np.array_equal(watermarked, image)
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
pytest tests/test_watermark.py -v
```

Expected: `ModuleNotFoundError: No module named 'watermark'`.

- [ ] **Step 3: Implement `watermark.py`**

```python
# watermark.py
import cv2
import numpy as np

_WATERMARK_TEXT = "AI GENERATED"


def add_disclosure_watermark(image_bgr: np.ndarray) -> np.ndarray:
    """Return a copy of image_bgr with a visible AI-disclosure watermark."""
    watermarked = image_bgr.copy()
    height, width = watermarked.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(width, height) / 400
    thickness = max(1, int(font_scale * 2))
    (text_width, text_height), _ = cv2.getTextSize(
        _WATERMARK_TEXT, font, font_scale, thickness
    )

    margin = 10
    bar_height = text_height + 2 * margin
    overlay = watermarked.copy()
    cv2.rectangle(overlay, (0, height - bar_height), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, watermarked, 0.4, 0, watermarked)

    text_x = max(margin, (width - text_width) // 2)
    text_y = height - margin - 2
    cv2.putText(
        watermarked,
        _WATERMARK_TEXT,
        (text_x, text_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return watermarked
```

- [ ] **Step 4: Run test to verify it passes**

```powershell
pytest tests/test_watermark.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add watermark.py tests/test_watermark.py
git commit -m "FR15: add mandatory AI-disclosure watermark module"
```

---

### Task 4: Flask `/swap` route

**Files:**
- Modify: `app.py`
- Modify: `tests/test_app.py`

**Interfaces:**
- Consumes: `swap_faces` (Task 2), `add_disclosure_watermark` (Task 3),
  `NoFaceDetectedError` (already imported from `face_detect`).
- Produces: route `POST /swap` returning JSON
  `{"swapped_image_png_base64": str}` on success (200), or `{"error": str}`
  on failure (400 missing images, 422 no face found, 500 encode failure).
  Task 5 (frontend) relies on exactly this field name.

- [ ] **Step 1: Add the failing tests to `tests/test_app.py`**

```python
def test_swap_route_rejects_missing_images(client):
    response = client.post("/swap", data={})
    assert response.status_code == 400
    assert "error" in response.get_json()


@REQUIRES_FIXTURE
def test_swap_route_returns_watermarked_image(client):
    with open(FIXTURE_PATH, "rb") as source_file, open(FIXTURE_PATH, "rb") as target_file:
        response = client.post(
            "/swap",
            data={
                "source": (source_file, "source.jpg"),
                "target": (target_file, "target.jpg"),
            },
            content_type="multipart/form-data",
        )
    assert response.status_code == 200
    body = response.get_json()
    assert "swapped_image_png_base64" in body
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
pytest tests/test_app.py -v
```

Expected: both new tests FAIL with 404 (no `/swap` route yet).

- [ ] **Step 3: Add the route to `app.py`**

Update the imports at the top:

```python
from face_detect import NoFaceDetectedError, detect_landmarks, estimate_head_pose
from face_swap import swap_faces
from watermark import add_disclosure_watermark
```

Add the new route (anywhere after the existing `/detect` route):

```python
@app.post("/swap")
def swap():
    source_file = request.files.get("source")
    target_file = request.files.get("target")
    if (
        source_file is None or source_file.filename == ""
        or target_file is None or target_file.filename == ""
    ):
        return jsonify(error="Both a source and a target image are required."), 400

    source_bytes = np.frombuffer(source_file.read(), dtype=np.uint8)
    source_bgr = cv2.imdecode(source_bytes, cv2.IMREAD_COLOR)
    target_bytes = np.frombuffer(target_file.read(), dtype=np.uint8)
    target_bgr = cv2.imdecode(target_bytes, cv2.IMREAD_COLOR)

    if source_bgr is None or target_bgr is None:
        return jsonify(error="Could not decode one of the images. Use PNG or JPEG."), 400

    try:
        swapped_bgr = swap_faces(source_bgr, target_bgr)
    except NoFaceDetectedError as exc:
        return jsonify(error=str(exc)), 422

    watermarked_bgr = add_disclosure_watermark(swapped_bgr)

    success, encoded = cv2.imencode(".png", watermarked_bgr)
    if not success:
        return jsonify(error="Failed to encode result image."), 500

    return jsonify(
        swapped_image_png_base64=base64.b64encode(encoded.tobytes()).decode("ascii"),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
pytest tests/test_app.py -v
```

Expected: all PASS or SKIP (none FAIL).

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_app.py
git commit -m "FR4-15: add /swap endpoint with mandatory watermark"
```

---

### Task 5: Frontend — face-swap form and result display

**Files:**
- Modify: `templates/index.html`
- Modify: `static/script.js`
- Modify: `static/style.css`

**Interfaces:**
- Consumes: `POST /swap` JSON shape from Task 4
  (`{swapped_image_png_base64}` or `{error}`).

- [ ] **Step 1: Add the swap section to `templates/index.html`**

Add this after the existing landmark-detection `</form>`/result block, before
`<script src="...">`:

```html
  <hr>
  <h2>Yüz Değiştirme</h2>
  <form id="swap-form">
    <label>Kaynak yüz: <input type="file" id="source-input" name="source" accept="image/png, image/jpeg" required></label><br>
    <label>Hedef görsel: <input type="file" id="target-input" name="target" accept="image/png, image/jpeg" required></label><br>
    <button type="submit">Değiştir</button>
  </form>
  <p id="swap-status"></p>
  <img id="swap-result-image" alt="" hidden>
```

- [ ] **Step 2: Add the handler to `static/script.js`**

Append to the end of the file:

```javascript
const swapForm = document.getElementById("swap-form");
const swapStatusEl = document.getElementById("swap-status");
const swapResultImage = document.getElementById("swap-result-image");

swapForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const sourceInput = document.getElementById("source-input");
  const targetInput = document.getElementById("target-input");
  if (!sourceInput.files.length || !targetInput.files.length) {
    return;
  }

  const formData = new FormData();
  formData.append("source", sourceInput.files[0]);
  formData.append("target", targetInput.files[0]);

  swapStatusEl.textContent = "İşleniyor...";
  swapResultImage.hidden = true;

  const response = await fetch("/swap", { method: "POST", body: formData });
  const body = await response.json();

  if (!response.ok) {
    swapStatusEl.textContent = `Hata: ${body.error}`;
    return;
  }

  swapStatusEl.textContent = "Yüz değiştirme tamamlandı.";
  swapResultImage.src = `data:image/png;base64,${body.swapped_image_png_base64}`;
  swapResultImage.hidden = false;
});
```

- [ ] **Step 3: Add spacing rule to `static/style.css`**

Append:

```css
hr {
  margin: 2rem 0;
}
```

- [ ] **Step 4: Manual browser test (required — this is a UI change)**

```powershell
python app.py
```

Open `http://127.0.0.1:5000/` and, in the new "Yüz Değiştirme" section:
1. Upload a source face photo and a different target photo → confirm a
   swapped result renders with the "AI GENERATED" watermark visible at the
   bottom.
2. Upload a source/target where one has no detectable face → confirm the
   status line shows the "No face detected" error, no crash.
3. Try submitting with only one of the two files selected → native
   `required` validation should block submission.

- [ ] **Step 5: Commit**

```bash
git add templates/index.html static/script.js static/style.css
git commit -m "FR4: add face-swap UI with before/after result display"
```
