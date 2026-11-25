# Phase 3 — Before/After Comparison & Quality Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show a before/after comparison for both existing features (landmark
detection and face swap) and surface objective quality/speed numbers:
processing time for both, plus a face-identity-similarity score for swap.
Satisfies FR13 and FR14 of the Topic 4 spec.

**Architecture:** No new modules. `face_swap.py` gains one pure function
(`compute_identity_similarity`); `app.py`'s existing `/detect` and `/swap`
routes gain timing and (for `/swap`) the similarity call, returned as new
JSON fields. The frontend shows the original file next to the server's
result using `URL.createObjectURL` — no round trip needed for the "before"
image.

**Tech Stack:** Same as Phase 1/2 (Flask, InsightFace, OpenCV); no new
dependencies.

**Spec:** `docs/superpowers/specs/2026-11-24-deepfake-interaction-system-design.md`
(§4, Phase 3). Builds on `docs/superpowers/plans/2026-11-24-phase2-face-swap.md`.

## Global Constraints

(From the design spec, unchanged from Phase 1/2, still binding)

- Uploaded/result face photos must never be committed to the repo — enforced
  by the existing root `.gitignore`.
- No model training — only pretrained libraries (InsightFace, MediaPipe) —
  spec §2.
- Commit after each task with a message referencing the functional
  requirement(s) it covers — spec §6.

## Review Focus

- No face detected in the swap result (should be rare, since the swap
  itself just produced a face, but a low-quality or heavily cropped result
  could fail re-detection) — `compute_identity_similarity` must return
  `None` rather than crash, and the `/swap` JSON response and the frontend
  must both handle a `null` similarity without erroring.
- A genuinely poor swap (very different source/target faces) can produce a
  *low or negative* similarity score — the UI must display it plainly as a
  number, not treat a low score as an error state.
- Submitting either form more than once in a session must not leak
  `URL.createObjectURL` object URLs — revoke the previous one before
  creating the next.
- `processing_time_seconds` must be a plain JSON-serializable float (not a
  numpy float type, which `jsonify` cannot serialize) for both routes.

---

### Task 1: Identity-similarity metric in `face_swap.py`

**Files:**
- Modify: `face_swap.py`
- Test: `tests/test_face_swap.py`

**Interfaces:**
- Consumes: `_get_analyzer()` (existing private helper in this file).
- Produces: `compute_identity_similarity(source_bgr: np.ndarray, result_bgr: np.ndarray) -> float | None`.
  Returns `None` if no face is found in `result_bgr`. Raises
  `NoFaceDetectedError` (already imported in this file) if no face is found
  in `source_bgr`. Task 3 (`app.py`) imports and calls this.

- [ ] **Step 1: Write the failing tests**

```python
def test_compute_identity_similarity_is_high_for_self_swap():
    image = cv2.imread(str(FIXTURE_PATH))
    result = swap_faces(image, image)
    similarity = compute_identity_similarity(image, result)
    assert similarity is not None
    assert similarity > 0.5


def test_compute_identity_similarity_returns_none_when_result_has_no_face():
    image = cv2.imread(str(FIXTURE_PATH)) if FIXTURE_PATH.exists() else None
    if image is None:
        pytest.skip("needs tests/fixtures/sample_face.jpg")
    blank = np.zeros((200, 200, 3), dtype="uint8")
    assert compute_identity_similarity(image, blank) is None
```

Add these under the existing `@REQUIRES_FIXTURE` marker in
`tests/test_face_swap.py`, and add `from face_swap import compute_identity_similarity`
to its imports.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_face_swap.py -v`
Expected: FAIL with `ImportError: cannot import name 'compute_identity_similarity'`.

- [ ] **Step 3: Implement `compute_identity_similarity(source_bgr: np.ndarray, result_bgr: np.ndarray) -> float | None` in `face_swap.py`**

Use `_get_analyzer().get(...)` on both images. If `source_bgr` has no face,
raise `NoFaceDetectedError("No face detected in the source image.")` (same
message style as `swap_faces`). If `result_bgr` has no face, return `None`.
Otherwise return `float(np.dot(source_face.normed_embedding, result_face.normed_embedding))`
— cosine similarity, since InsightFace embeddings are already unit-norm.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_face_swap.py -v`
Expected: all PASS or SKIP (none FAIL).

- [ ] **Step 5: Commit**

```bash
git add face_swap.py tests/test_face_swap.py
git commit -m "FR14: add identity-similarity metric for face swap"
```

---

### Task 2: Timing and metrics in the `/detect` and `/swap` routes

**Files:**
- Modify: `app.py`
- Modify: `tests/test_app.py`

**Interfaces:**
- Consumes: `compute_identity_similarity` (Task 1).
- Produces: `/detect`'s success JSON gains `processing_time_seconds: float`.
  `/swap`'s success JSON gains `processing_time_seconds: float` and
  `identity_similarity: float | None`. Task 3 (frontend) relies on exactly
  these field names and on `identity_similarity` possibly being `null`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_app.py`, extending the existing fixture-gated tests:

```python
@REQUIRES_FIXTURE
def test_detect_route_includes_processing_time(client):
    with open(FIXTURE_PATH, "rb") as image_file:
        response = client.post(
            "/detect",
            data={"image": (image_file, "sample_face.jpg")},
            content_type="multipart/form-data",
        )
    body = response.get_json()
    assert isinstance(body["processing_time_seconds"], float)


@REQUIRES_FIXTURE
def test_swap_route_includes_processing_time_and_similarity(client):
    with open(FIXTURE_PATH, "rb") as source_file, open(FIXTURE_PATH, "rb") as target_file:
        response = client.post(
            "/swap",
            data={
                "source": (source_file, "source.jpg"),
                "target": (target_file, "target.jpg"),
            },
            content_type="multipart/form-data",
        )
    body = response.get_json()
    assert isinstance(body["processing_time_seconds"], float)
    assert isinstance(body["identity_similarity"], float)
    assert body["identity_similarity"] > 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_app.py -v`
Expected: both new tests FAIL with `KeyError: 'processing_time_seconds'`.

- [ ] **Step 3: Add timing to `/detect` in `app.py`**

Import `time` at the top of the file. Wrap the existing
`detect_landmarks(...)` + `estimate_head_pose(...)` calls with
`time.perf_counter()` before and after (place the end timer after
`estimate_head_pose` returns, before encoding), and add
`processing_time_seconds=round(elapsed, 3)` to the existing `jsonify(...)`
call's keyword arguments. `round()` on a Python float stays a plain float,
satisfying the Review Focus item about JSON-serializable types.

- [ ] **Step 4: Add timing and similarity to `/swap` in `app.py`**

Import `compute_identity_similarity` from `face_swap` alongside the existing
`swap_faces` import. Time the `swap_faces(...)` call the same way as Task 2
Step 3. After it succeeds, call
`identity_similarity = compute_identity_similarity(source_bgr, swapped_bgr)`
(before watermarking — the watermark bar would otherwise sit over part of
the face the analyzer needs). Add both
`processing_time_seconds=round(elapsed, 3)` and
`identity_similarity=identity_similarity` to the existing `jsonify(...)`
call's keyword arguments (Flask's `jsonify` serializes `None` as JSON
`null` natively, satisfying the Review Focus item about a missing face in
the result).

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_app.py -v`
Expected: all PASS or SKIP (none FAIL).

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_app.py
git commit -m "FR13-14: add processing-time and identity-similarity metrics to API responses"
```

---

### Task 3: Before/after display and metrics in the frontend

**Files:**
- Modify: `templates/index.html`
- Modify: `static/script.js`
- Modify: `static/style.css`

**Interfaces:**
- Consumes: `/detect` and `/swap` JSON shapes from Task 2, including the new
  `processing_time_seconds` field on both and `identity_similarity`
  (possibly `null`) on `/swap`.

- [ ] **Step 1: Add "before" image elements in `templates/index.html`**

Wrap each feature's existing result `<img>` together with a new "before"
`<img>` in a `<div class="before-after">` container holding two labelled
sub-sections ("Önce" / "Sonra"). Add `id="original-image"` for the landmark
section's before-image and `id="swap-original-image"` for the swap
section's before-image (the target photo, not the source — FR13 compares
the original being transformed against the generated result).

- [ ] **Step 2: Show the "before" image and revoke prior object URLs in `static/script.js`**

In each form's submit handler, before building `FormData`: if a
module-level variable holding the previous object URL for that image is
set, call `URL.revokeObjectURL` on it (Review Focus: avoid leaking object
URLs across repeated submissions); then call
`URL.createObjectURL(file)`, store the new URL in that variable, and set it
as the corresponding before-image's `src`, clearing `hidden`.

- [ ] **Step 3: Display the new metrics in both success handlers**

In the `/detect` success branch, append to `statusEl.textContent`:
`` ` (${body.processing_time_seconds.toFixed(2)} sn)` ``.

In the `/swap` success branch, append to `swapStatusEl.textContent` the
processing time the same way, and the similarity — guard for `null`:

```javascript
const similarityText = body.identity_similarity === null
  ? "yüz tespit edilemedi"
  : `${(body.identity_similarity * 100).toFixed(1)}%`;
```

then append `` ` (${elapsedText}, benzerlik: ${similarityText})` `` to
`swapStatusEl.textContent`. This satisfies the Review Focus item that a low
or negative similarity must still render as a plain number
(`(-0.12 * 100).toFixed(1)` → `"-12.0"`, no exception).

- [ ] **Step 4: Add layout rules to `static/style.css`**

```css
.before-after {
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
  margin-top: 1rem;
}

.before-after figure {
  margin: 0;
}

.before-after img {
  max-width: 100%;
  width: 300px;
  border: 1px solid #ccc;
  display: block;
}
```

Remove the now-redundant standalone `#result-image, #swap-result-image`
width/border rule, since `.before-after img` covers both before- and
after-images.

- [ ] **Step 5: Manual browser test (required — this is a UI change)**

```powershell
python app.py
```

Open `http://127.0.0.1:5000/` and:
1. Landmark section: upload a real face photo → confirm "Önce" shows the
   original and "Sonra" shows the landmark overlay, with a processing-time
   reading in the status line.
2. Swap section: upload two different real face photos → confirm "Önce"
   shows the target photo and "Sonra" shows the watermarked swap, with
   processing time and a similarity percentage in the status line.
3. Swap section again, with a source/target pair where one has no face →
   confirm the existing error handling from Phase 2 still works unchanged.
4. Submit the swap form twice in a row with different files → confirm the
   "Önce" image updates correctly both times (no stale image, no console
   errors about revoked object URLs).

- [ ] **Step 6: Commit**

```bash
git add templates/index.html static/script.js static/style.css
git commit -m "FR13: add before/after comparison view with time/similarity metrics"
```
