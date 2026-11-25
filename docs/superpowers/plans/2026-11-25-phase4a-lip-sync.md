# Phase 4a — Lip Sync (Wav2Lip) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "03 — Dudak Senkronu" tab: upload a video + a separate audio
file, get back the video with its mouth re-synced to that audio
(Wav2Lip), watermarked the same way every other feature is. Satisfies FR6
and FR12 of the Topic 4 spec.

**Architecture:** Wav2Lip's code and its own virtual environment live
outside the Flask app's `.venv`, in a git-ignored `vendor/wav2lip/`
directory, set up once per machine by `scripts/setup_lipsync.py`. The
Flask app's new `lip_sync.py` module calls that separate environment via
`subprocess`, then burns in the disclosure watermark with an `ffmpeg`
filter. A new `/lipsync` route saves the two uploads, calls that module,
and returns a URL to the result; a second route serves the result file.
The frontend adds a third tab with `<video>` elements in place of the
`<img>`s the other two tabs use.

**Tech Stack:** Wav2Lip (vendored, not a dependency of this project's own
`requirements.txt`), its own venv with `torch` (CUDA build),
`librosa`, `opencv-contrib-python`, `scipy`, `tqdm`; `ffmpeg` as a system
binary; Flask, `subprocess`.

**Spec:** `docs/superpowers/specs/2026-11-25-lip-sync-design.md`. Builds on
`docs/superpowers/specs/2026-11-24-deepfake-interaction-system-design.md`.

## Global Constraints

(From the design doc, §3a/§3b/§4 — binding for every task below)

- `ffmpeg` must be on `PATH` — a new system-level dependency, not a pip
  package (spec §4).
- Wav2Lip's code is never committed to this project's public repo — it
  lives in git-ignored `vendor/wav2lip/`, cloned at a pinned commit
  (`bac9a81e63ecc153202353372e5724b83d9e6322` of
  `https://github.com/Rudrabha/Wav2Lip`, verified in this session — spec
  §3a/§5).
- Wav2Lip runs in its **own** virtual environment, invoked via
  `subprocess` — never imported into the main Flask app's `.venv` (spec
  §3b, for both crash isolation and avoiding dependency collisions with
  mediapipe/insightface's pinned versions already in `requirements.txt`).
- Every result video carries the same visible "AI GENERATED" disclosure
  the image features already apply (project-wide FR15).
- `app.run(...)` gains `threaded=True` so a long-running lip-sync request
  can't freeze the detect/swap tabs for another concurrent user (spec
  §3b; this is also the anti-goal: a lip-sync failure or slowness must
  never break the other two tabs).
- Commit after each task with a message referencing the functional
  requirement(s) it covers.

## Review Focus

- **Wav2Lip not yet set up on this machine** (fresh clone, `setup_lipsync.py`
  never run) — the `/lipsync` route must return a clear, actionable error
  (not a raw `FileNotFoundError` traceback) telling the user to run the
  setup script.
- **A video with no detectable face** (Wav2Lip's own face detector finds
  nothing) — the vendored `inference.py` exits non-zero in this case; the
  route must turn that into a controlled error response, not an unhandled
  exception that takes down the request (or, per the anti-goal, risks
  the whole app).
- **Path traversal on the result-serving route** — `GET
  /lipsync/result/<filename>` must not be tricked into serving an
  arbitrary file via a crafted `filename` (e.g. `../app.py`).
- **Upload cleanup on failure** — the temporary video/audio files saved to
  `uploads/` must be deleted even when processing raises, not just on the
  success path (a `finally` block, not an afterthought at the end of the
  happy path).
- **`MAX_CONTENT_LENGTH` must actually admit a valid-sized request** — the
  existing 10 MB cap (set in Phase 1 for single images) is too small for
  a video+audio pair; raising it is easy to do and easy to forget to
  verify.

---

### Task 1: Vendor setup script for Wav2Lip

**Files:**
- Create: `scripts/setup_lipsync.py`
- Modify: `.gitignore`
- Modify: `models/README.md`

**Interfaces:**
- Produces, once run: `vendor/wav2lip/Wav2Lip/` (patched clone),
  `vendor/wav2lip/Wav2Lip/checkpoints/wav2lip_gan.pth`,
  `vendor/wav2lip/.venv/` (with the packages listed in Global
  Constraints installed). Task 2 (`lip_sync.py`) depends on these three
  paths existing at exactly these locations relative to the project
  root.

- [ ] **Step 1: Add `vendor/` to `.gitignore`**

```
# Wav2Lip vendored clone + its own venv (see models/README.md) — not
# committed, has no declared license and isn't ours to redistribute.
vendor/
```

- [ ] **Step 2: Write `scripts/setup_lipsync.py`**

```python
"""One-time setup for the lip-sync feature (Wav2Lip).

Run once per machine: `python scripts/setup_lipsync.py`
Safe to re-run — every step is idempotent (skips what's already done).

Needs `git` and `ffmpeg` on PATH; this script does not install those.
"""
import pathlib
import subprocess
import sys
import urllib.request

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
VENDOR_DIR = PROJECT_ROOT / "vendor" / "wav2lip"
REPO_DIR = VENDOR_DIR / "Wav2Lip"
VENV_DIR = VENDOR_DIR / ".venv"
VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"
PINNED_COMMIT = "bac9a81e63ecc153202353372e5724b83d9e6322"
CHECKPOINT_PATH = REPO_DIR / "checkpoints" / "wav2lip_gan.pth"
CHECKPOINT_URL = "https://huggingface.co/Nekochu/Wav2Lip/resolve/main/wav2lip_gan.pth"


def clone_repo() -> None:
    if REPO_DIR.exists():
        print(f"{REPO_DIR} already exists, skipping clone.")
        return
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "https://github.com/Rudrabha/Wav2Lip.git", str(REPO_DIR)],
        check=True,
    )
    subprocess.run(["git", "checkout", PINNED_COMMIT], cwd=str(REPO_DIR), check=True)


def apply_patches() -> None:
    _patch_file(
        REPO_DIR / "audio.py",
        "librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels,",
        "librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels,",
    )
    _patch_file(
        REPO_DIR / "face_detection" / "utils.py",
        "dtype=np.int)",
        "dtype=np.int32)",
    )


def _patch_file(path: pathlib.Path, old: str, new: str) -> None:
    content = path.read_text()
    if new in content:
        print(f"{path.name} already patched, skipping.")
        return
    if old not in content:
        raise RuntimeError(f"Expected text not found in {path} — patch needs updating.")
    path.write_text(content.replace(old, new))
    print(f"Patched {path}.")


def create_venv() -> None:
    if VENV_PYTHON.exists():
        print(f"{VENV_DIR} already exists, skipping venv creation.")
        return
    subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)


def install_dependencies() -> None:
    subprocess.run(
        [
            str(VENV_PYTHON), "-m", "pip", "install",
            "torch==2.10.0+cu128", "--index-url", "https://download.pytorch.org/whl/cu128",
        ],
        check=True,
    )
    subprocess.run(
        [
            str(VENV_PYTHON), "-m", "pip", "install",
            "librosa==1.0.1", "numpy==2.5.5", "opencv-contrib-python==5.1.0.95",
            "tqdm==4.70.1", "scipy==1.18.1",
        ],
        check=True,
    )


def download_checkpoint() -> None:
    if CHECKPOINT_PATH.exists():
        print(f"{CHECKPOINT_PATH} already present, skipping.")
        return
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading wav2lip_gan.pth (~415 MB) ...")
    urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT_PATH)
    print(f"Saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    clone_repo()
    apply_patches()
    create_venv()
    install_dependencies()
    download_checkpoint()
    print("Lip-sync setup complete.")
```

- [ ] **Step 3: Run it and verify**

```powershell
python scripts/setup_lipsync.py
```

Expected: completes with `Lip-sync setup complete.`, and
`vendor/wav2lip/Wav2Lip/checkpoints/wav2lip_gan.pth` exists
(~415 MB).

Then verify the CUDA build actually installed correctly:

```powershell
vendor\wav2lip\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

Expected: `True` on a machine with an NVIDIA GPU (both this project's
machines have one). If `False` on the laptop, the CUDA wheel didn't match
that machine's setup — fall back there with:
`vendor\wav2lip\.venv\Scripts\python.exe -m pip install torch==2.10.0`
(CPU-only; Wav2Lip's own code already auto-detects and falls back, just
slower).

- [ ] **Step 4: Document the setup in `models/README.md`**

Add a new section:

```markdown
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
```

- [ ] **Step 5: Commit**

```bash
git add scripts/setup_lipsync.py .gitignore models/README.md
git commit -m "FR6 setup: add Wav2Lip vendoring/setup script"
```

---

### Task 2: `lip_sync.py` orchestration module

**Files:**
- Create: `lip_sync.py`
- Test: `tests/test_lip_sync.py`

**Interfaces:**
- Produces: `run_lip_sync(video_path: pathlib.Path, audio_path: pathlib.Path, output_path: pathlib.Path) -> None`.
  Writes the final, watermarked result to `output_path`. Raises
  `LipSyncUnavailableError` (defined in this module) if Wav2Lip's vendor
  venv/checkpoint isn't set up. Raises `RuntimeError` if the vendored
  Wav2Lip subprocess itself fails (e.g. no face found). Task 3 (`app.py`)
  imports `run_lip_sync` and `LipSyncUnavailableError`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_lip_sync.py
import pathlib

import pytest

from lip_sync import LipSyncUnavailableError, run_lip_sync

FIXTURE_VIDEO = pathlib.Path(__file__).parent / "fixtures" / "sample_talking.mp4"
FIXTURE_AUDIO = pathlib.Path(__file__).parent / "fixtures" / "sample_speech.wav"
REQUIRES_WAV2LIP = pytest.mark.skipif(
    not (pathlib.Path(__file__).parent.parent / "vendor" / "wav2lip" / "Wav2Lip"
         / "checkpoints" / "wav2lip_gan.pth").exists(),
    reason="Run `python scripts/setup_lipsync.py` first.",
)
REQUIRES_FIXTURES = pytest.mark.skipif(
    not (FIXTURE_VIDEO.exists() and FIXTURE_AUDIO.exists()),
    reason=(
        "Add a short talking-face video at tests/fixtures/sample_talking.mp4 "
        "and a short speech clip at tests/fixtures/sample_speech.wav "
        "(both gitignored, add them locally)."
    ),
)


def test_run_lip_sync_raises_when_wav2lip_not_set_up(tmp_path, monkeypatch):
    import lip_sync

    monkeypatch.setattr(
        lip_sync, "_CHECKPOINT_PATH", tmp_path / "does-not-exist.pth"
    )
    with pytest.raises(LipSyncUnavailableError):
        run_lip_sync(tmp_path / "video.mp4", tmp_path / "audio.wav", tmp_path / "out.mp4")


def test_run_lip_sync_raises_runtime_error_when_inference_fails(tmp_path, monkeypatch):
    import subprocess

    import lip_sync

    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"fake")
    fake_venv_python = tmp_path / "python.exe"
    fake_venv_python.write_bytes(b"fake")
    monkeypatch.setattr(lip_sync, "_CHECKPOINT_PATH", checkpoint)
    monkeypatch.setattr(lip_sync, "_VENV_PYTHON", fake_venv_python)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, returncode=1, stdout="", stderr="no face detected"),
    )

    with pytest.raises(RuntimeError):
        run_lip_sync(tmp_path / "video.mp4", tmp_path / "audio.wav", tmp_path / "out.mp4")


@REQUIRES_WAV2LIP
@REQUIRES_FIXTURES
def test_run_lip_sync_produces_watermarked_output(tmp_path):
    output_path = tmp_path / "result.mp4"
    run_lip_sync(FIXTURE_VIDEO, FIXTURE_AUDIO, output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_lip_sync.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lip_sync'`.

- [ ] **Step 3: Implement `lip_sync.py`**

```python
import pathlib
import subprocess

_VENDOR_DIR = pathlib.Path(__file__).parent / "vendor" / "wav2lip"
_REPO_DIR = _VENDOR_DIR / "Wav2Lip"
_VENV_PYTHON = _VENDOR_DIR / ".venv" / "Scripts" / "python.exe"
_CHECKPOINT_PATH = _REPO_DIR / "checkpoints" / "wav2lip_gan.pth"

# fontfile is required: ffmpeg's drawtext falls back to fontconfig when
# it's omitted, and fontconfig has no config file on a stock Windows
# install, which crashes ffmpeg outright rather than erroring cleanly.
_WATERMARK_FILTER = (
    "drawtext=text='AI GENERATED':fontfile='C\\:/Windows/Fonts/arial.ttf':"
    "fontcolor=white:fontsize=24:box=1:boxcolor=black@0.6:"
    "x=(w-text_w)/2:y=h-th-20"
)


class LipSyncUnavailableError(Exception):
    """Raised when the vendored Wav2Lip environment isn't set up yet."""


def run_lip_sync(
    video_path: pathlib.Path, audio_path: pathlib.Path, output_path: pathlib.Path
) -> None:
    """Re-sync video_path's mouth to audio_path, writing a watermarked
    result to output_path.

    Raises LipSyncUnavailableError if the vendored Wav2Lip setup (see
    scripts/setup_lipsync.py) hasn't been run on this machine. Raises
    RuntimeError if the vendored inference process itself fails (for
    example, no face found in video_path).
    """
    if not _VENV_PYTHON.exists() or not _CHECKPOINT_PATH.exists():
        raise LipSyncUnavailableError(
            "Lip sync isn't set up on this machine yet. "
            "Run `python scripts/setup_lipsync.py` first."
        )

    raw_output = output_path.with_name(output_path.stem + "_raw.mp4")
    result = subprocess.run(
        [
            str(_VENV_PYTHON), "inference.py",
            "--checkpoint_path", str(_CHECKPOINT_PATH),
            "--face", str(video_path.resolve()),
            "--audio", str(audio_path.resolve()),
            "--outfile", str(raw_output.resolve()),
        ],
        cwd=str(_REPO_DIR),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Wav2Lip inference failed:\n{result.stderr[-2000:]}")

    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(raw_output),
                "-vf", _WATERMARK_FILTER,
                "-codec:a", "copy",
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )
    finally:
        raw_output.unlink(missing_ok=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_lip_sync.py -v`
Expected: `test_run_lip_sync_raises_when_wav2lip_not_set_up` and
`test_run_lip_sync_raises_runtime_error_when_inference_fails` PASS.
`test_run_lip_sync_produces_watermarked_output` PASSES or SKIPS
(depending on whether setup/fixtures are present on this machine — never
FAILS).

- [ ] **Step 5: Document the fixture requirement**

Add to `tests/fixtures/README.md` (append to the existing file):

```markdown

For lip-sync tests: a few-seconds video with a visible talking face at
`sample_talking.mp4`, and a short speech/audio clip at
`sample_speech.wav`. Also gitignored — add them locally to run the
fixture-gated lip-sync tests.
```

- [ ] **Step 6: Commit**

```bash
git add lip_sync.py tests/test_lip_sync.py tests/fixtures/README.md
git commit -m "FR6: add lip_sync.py orchestrating vendored Wav2Lip via subprocess"
```

---

### Task 3: Flask `/lipsync` routes

**Files:**
- Modify: `app.py`
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: `run_lip_sync`, `LipSyncUnavailableError` from `lip_sync.py`
  (Task 2).
- Produces: `POST /lipsync` — accepts multipart fields `video` and
  `audio`; success (200) returns
  `{"result_url": str, "processing_time_seconds": float}`; failure
  returns `{"error": str}` with 400 (missing files), 503 (Wav2Lip not set
  up), or 500 (processing failed). `GET /lipsync/result/<filename>`
  serves the file from the `outputs/` directory. Task 4 (frontend) relies
  on exactly these field names and the result being a URL, not inline
  base64.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_app.py`:

```python
def test_lipsync_route_rejects_missing_files(client):
    response = client.post("/lipsync", data={})
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_lipsync_route_returns_503_when_wav2lip_not_set_up(client, monkeypatch, tmp_path):
    import lip_sync

    monkeypatch.setattr(
        lip_sync, "_CHECKPOINT_PATH", tmp_path / "does-not-exist.pth"
    )
    data = {
        "video": (io.BytesIO(b"fake video bytes"), "video.mp4"),
        "audio": (io.BytesIO(b"fake audio bytes"), "audio.wav"),
    }
    response = client.post("/lipsync", data=data, content_type="multipart/form-data")
    assert response.status_code == 503
    assert "error" in response.get_json()


def test_lipsync_route_cleans_up_uploads_after_failure(client, monkeypatch, tmp_path):
    import lip_sync
    from app import UPLOADS_DIR

    monkeypatch.setattr(
        lip_sync, "_CHECKPOINT_PATH", tmp_path / "does-not-exist.pth"
    )
    data = {
        "video": (io.BytesIO(b"fake video bytes"), "video.mp4"),
        "audio": (io.BytesIO(b"fake audio bytes"), "audio.wav"),
    }
    client.post("/lipsync", data=data, content_type="multipart/form-data")

    assert list(UPLOADS_DIR.glob("*")) == []


def test_lipsync_route_returns_500_when_processing_fails(client, monkeypatch):
    import app as app_module

    def fake_run_lip_sync(video_path, audio_path, output_path):
        raise RuntimeError("no face detected")

    monkeypatch.setattr(app_module, "run_lip_sync", fake_run_lip_sync)

    data = {
        "video": (io.BytesIO(b"fake video bytes"), "video.mp4"),
        "audio": (io.BytesIO(b"fake audio bytes"), "audio.wav"),
    }
    response = client.post("/lipsync", data=data, content_type="multipart/form-data")
    assert response.status_code == 500
    assert "error" in response.get_json()


def test_lipsync_result_route_rejects_path_traversal(client):
    response = client.get("/lipsync/result/..%2F..%2Fapp.py")
    assert response.status_code == 404


def test_max_content_length_admits_a_realistic_video_and_audio_pair():
    from app import app as flask_app

    # 20 MB video + 10 MB audio, per the spec's size guidance — must fit.
    assert flask_app.config["MAX_CONTENT_LENGTH"] >= 30 * 1024 * 1024
```

Add `import io` to the top of `tests/test_app.py` if not already present.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_app.py -v`
Expected: all six new tests FAIL — `/lipsync` doesn't exist yet (404
instead of 400/503/500), and `MAX_CONTENT_LENGTH` is still the old 10 MB.

- [ ] **Step 3: Add the routes to `app.py`**

Update imports:

```python
import pathlib
import re
import uuid

from flask import send_from_directory
from lip_sync import LipSyncUnavailableError, run_lip_sync
```

Raise the upload cap (replace the existing `MAX_CONTENT_LENGTH` line):

```python
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024  # 40 MB (lip-sync needs more than a single image)
```

Add directory constants near the top of the file:

```python
UPLOADS_DIR = pathlib.Path("uploads")
OUTPUTS_DIR = pathlib.Path("outputs")
```

Add the routes:

```python
@app.post("/lipsync")
def lipsync():
    video_file = request.files.get("video")
    audio_file = request.files.get("audio")
    if (
        video_file is None or video_file.filename == ""
        or audio_file is None or audio_file.filename == ""
    ):
        return jsonify(error="Both a video and an audio file are required."), 400

    UPLOADS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)

    token = uuid.uuid4().hex
    video_path = UPLOADS_DIR / f"{token}_video.mp4"
    audio_path = UPLOADS_DIR / f"{token}_audio.wav"
    output_path = OUTPUTS_DIR / f"{token}.mp4"
    video_file.save(video_path)
    audio_file.save(audio_path)

    try:
        start = time.perf_counter()
        try:
            run_lip_sync(video_path, audio_path, output_path)
        except LipSyncUnavailableError as exc:
            return jsonify(error=str(exc)), 503
        except RuntimeError:
            return jsonify(error="Lip sync processing failed for this video/audio pair."), 500
        elapsed = time.perf_counter() - start
    finally:
        video_path.unlink(missing_ok=True)
        audio_path.unlink(missing_ok=True)

    return jsonify(
        result_url=f"/lipsync/result/{token}.mp4",
        processing_time_seconds=round(elapsed, 3),
    )


@app.get("/lipsync/result/<filename>")
def lipsync_result(filename):
    if not re.fullmatch(r"[0-9a-f]{32}\.mp4", filename):
        return jsonify(error="Not found."), 404
    return send_from_directory(OUTPUTS_DIR, filename)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_app.py -v`
Expected: all PASS (none SKIP — these six don't need the fixture/vendor
markers, they exercise only the route logic).

- [ ] **Step 5: Add `threaded=True` to the dev server**

```python
if __name__ == "__main__":
    app.run(debug=True, threaded=True)
```

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_app.py
git commit -m "FR6-15: add /lipsync routes, raise upload cap, enable threaded dev server"
```

---

### Task 4: Frontend — lip-sync tab

**Files:**
- Modify: `templates/index.html`
- Modify: `static/script.js`
- Modify: `static/style.css`

**Interfaces:**
- Consumes: `POST /lipsync` JSON shape from Task 3
  (`{result_url, processing_time_seconds}` or `{error}`).

- [ ] **Step 1: Add the third tab button to `templates/index.html`**

In the `<nav class="module-tabs">` block, after the existing two buttons:

```html
<button type="button" class="tab-button" id="tab-lipsync" role="tab" aria-selected="false" aria-controls="panel-lipsync">03 — Dudak Senkronu</button>
```

- [ ] **Step 2: Add the panel markup**

After the `panel-swap` `</section>`, before `</main>`:

```html
<section class="module" id="panel-lipsync" role="tabpanel" aria-labelledby="tab-lipsync module-3-title" hidden>
  <div class="module-head">
    <h2 id="module-3-title">Dudak Senkronu</h2>
  </div>

  <form id="lipsync-form" class="module-form">
    <label>
      Video (konuşan bir yüz)
      <input type="file" id="lipsync-video-input" name="video" accept="video/mp4" required>
    </label>
    <label>
      Ses dosyası
      <input type="file" id="lipsync-audio-input" name="audio" accept="audio/wav, audio/mpeg" required>
    </label>
    <div class="form-actions">
      <button type="submit">Senkronize Et</button>
    </div>
  </form>

  <p id="lipsync-status" class="status-line"></p>
  <dl id="lipsync-readout" class="readout" hidden></dl>

  <div class="before-after">
    <figure>
      <div class="frame" id="lipsync-original-frame"><video id="lipsync-original-video" controls hidden></video></div>
      <figcaption>Önce</figcaption>
    </figure>
    <figure>
      <div class="frame" id="lipsync-result-frame"><video id="lipsync-result-video" controls hidden></video></div>
      <figcaption>Sonra</figcaption>
    </figure>
  </div>
</section>
```

- [ ] **Step 3: Register the new tab/panel pair in `static/script.js`**

In the existing `tabPanels` object, add:

```javascript
"tab-lipsync": document.getElementById("panel-lipsync"),
```

(`tabButtons` already queries `.tab-button` generically, so the new
button is picked up automatically — no change needed there.)

- [ ] **Step 4: Add the form handler**

Append to `static/script.js`:

```javascript
const lipsyncForm = document.getElementById("lipsync-form");
const lipsyncStatusEl = document.getElementById("lipsync-status");
const lipsyncReadoutEl = document.getElementById("lipsync-readout");
const lipsyncOriginalFrame = document.getElementById("lipsync-original-frame");
const lipsyncOriginalVideo = document.getElementById("lipsync-original-video");
const lipsyncResultFrame = document.getElementById("lipsync-result-frame");
const lipsyncResultVideo = document.getElementById("lipsync-result-video");
let lipsyncOriginalObjectUrl = null;

function showVideo(videoEl, frameEl, src) {
  videoEl.src = src;
  videoEl.hidden = false;
  frameEl.classList.add("has-image");
}

lipsyncForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const videoInput = document.getElementById("lipsync-video-input");
  const audioInput = document.getElementById("lipsync-audio-input");
  if (!videoInput.files.length || !audioInput.files.length) {
    return;
  }
  const videoFile = videoInput.files[0];

  if (lipsyncOriginalObjectUrl) {
    URL.revokeObjectURL(lipsyncOriginalObjectUrl);
  }
  lipsyncOriginalObjectUrl = URL.createObjectURL(videoFile);
  showVideo(lipsyncOriginalVideo, lipsyncOriginalFrame, lipsyncOriginalObjectUrl);

  const formData = new FormData();
  formData.append("video", videoFile);
  formData.append("audio", audioInput.files[0]);

  lipsyncStatusEl.textContent = "İşleniyor... bu biraz uzun sürebilir.";
  setReadout(lipsyncReadoutEl, []);
  lipsyncResultVideo.hidden = true;
  lipsyncResultFrame.classList.remove("has-image");

  let response;
  let body;
  try {
    response = await fetch("/lipsync", { method: "POST", body: formData });
    body = await response.json();
  } catch (err) {
    lipsyncStatusEl.textContent = "Hata: sunucudan beklenmeyen bir yanıt geldi.";
    return;
  }

  if (!response.ok) {
    lipsyncStatusEl.textContent = `Hata: ${body.error}`;
    return;
  }

  lipsyncStatusEl.textContent = "Dudak senkronu tamamlandı.";
  setReadout(lipsyncReadoutEl, [
    ["SÜRE", `${body.processing_time_seconds.toFixed(2)}s`],
  ]);
  showVideo(lipsyncResultVideo, lipsyncResultFrame, body.result_url);
});
```

- [ ] **Step 5: Add a `.frame video` rule to `static/style.css`**

Alongside the existing `.frame img` rule:

```css
.frame video {
  width: 100%;
  height: 100%;
  object-fit: contain;
}
```

- [ ] **Step 6: Manual browser test (required — this is a UI change)**

```powershell
python app.py
```

Open `http://127.0.0.1:5000/`, switch to "03 — Dudak Senkronu":
1. Upload a short talking-face video + a short audio clip → confirm
   "Önce" plays the original video, status shows "İşleniyor..." then,
   after it finishes, "Sonra" plays the result with the mouth re-synced
   and the "AI GENERATED" watermark burned in, plus a processing-time
   reading.
2. Switch to "01 — Tespit" or "02 — Değiştirme" **while the lip-sync
   request from step 1 is still processing** (open a second browser tab
   to the same URL, or fire the two requests close together) → confirm
   the other tab's request still completes instead of hanging
   (`threaded=True` working).
3. Try submitting with no files selected → native `required` validation
   blocks it, same as the other two forms.

- [ ] **Step 7: Commit**

```bash
git add templates/index.html static/script.js static/style.css
git commit -m "FR6: add lip-sync tab with video before/after and processing-time readout"
```
