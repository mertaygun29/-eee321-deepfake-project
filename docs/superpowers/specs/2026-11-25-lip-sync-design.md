# Phase 4a — Lip Sync (Audio-Driven Dubbing) — Design

**Date:** 2026-11-25
**Course:** EEE 321 DSP Project (Topic 4)
**Builds on:** `docs/superpowers/specs/2026-11-24-deepfake-interaction-system-design.md`

## 1. Intent

Add lip-sync to the existing Flask app: upload a video of someone talking
plus a separate audio file, and get back the same video with the mouth
re-synced to the new audio (a dubbing scenario). This satisfies FR6 ("the
system shall provide lip synchronization, modifying mouth movements to
match supplied or synthesized speech") and FR12 ("audio-visual
synchronization") from the Topic 4 requirements. The instructor flagged
FR10 (real-time operation) as also required eventually, but explicitly
agreed lip-sync comes first — FR10 is a separate, later piece of work, not
bundled into this one.

## 2. Goals / Non-goals / Anti-goals

**Goals:**
- A working "03 — Dudak Senkronu" tab, consistent with the existing
  tab-based UI, where a user uploads a video + an audio file and gets a
  lip-synced result video back, playable in the browser.
- The result carries the same visible "AI GENERATED" disclosure the image
  features already apply (FR15 is project-wide, not feature-specific).
- Setup instructions that work on both the desktop and the friend's
  laptop, following the existing `scripts/download_models.py` /
  `models/README.md` pattern where it still fits.

**Non-goals (this phase):**
- Real-time / live operation (FR10) — separate future phase.
- A single-photo-to-talking-video mode — out of scope; input is always an
  existing video with a real face in it.
- An automated lip-sync *quality* score (e.g. a SyncNet-based metric) —
  processing time is reported (consistent with FR14 elsewhere in the app);
  a quality score is a nice-to-have for later, not required now.

**Anti-goals:**
- This feature must not be able to break the already-working landmark and
  face-swap tabs. It gets its own module/route; a failure here (missing
  ffmpeg, a bad upload) must degrade to a visible error in its own tab,
  never crash the whole app.

## 3. Approaches considered

Two real alternatives were evaluated before settling on this design:

- **MuseTalk (Tencent).** Better visual quality, but investigation found
  it needs `mmpose` (a separate, historically fragile-to-install
  framework, not even listed in its own `requirements.txt`), a pinned
  `tensorflow==2.12.0` that has no stable wheel for our Python 3.14 setup
  (only a release candidate exists), and six model files from four
  different hosts, including a Google Drive `gdown` link (known to be
  unreliable for larger files). Rejected: too much installation risk for
  the time budget and for reproducing setup on a second machine.
- **Wav2Lip (chosen).** Older, lower peak quality, but a small, well
  understood dependency surface (numpy, scipy, opencv, torch, librosa,
  tqdm) and a single checkpoint file. One known, well-documented
  compatibility fix is needed (a `librosa.filters.mel(...)` call uses
  positional args that modern librosa made keyword-only — a one-line
  patch). Weights are available from a working HuggingFace mirror; the
  bundled face detector's weights auto-download via `torch.hub` from a
  long-stable host. Chosen for reliability: this project has consistently
  favored "works predictably on both machines" over peak quality (the
  same reasoning that picked CPU-only InsightFace over a GPU/CPU-provider
  branch in Phase 2).

## 3a. Verified during research (not just assumed)

A fresh-context review of this document correctly flagged several claims
above as things a builder would need re-confirmed. They were, in fact,
already checked in this session, not just assumed:

- **torch has a working Python 3.14 wheel.** `pip download torch --no-deps`
  resolved and downloaded `torch-2.15.0-cp314-cp314-win_amd64.whl`
  successfully in this project's actual `.venv` (Python 3.14.6). This is
  what makes Wav2Lip viable where MuseTalk's `tensorflow==2.12.0` wasn't —
  the comparison in §3 is apples-to-apples, not one side checked and the
  other assumed.
- **The canonical `Rudrabha/Wav2Lip` repo's face-detector download path
  works today.** Its `face_detection/detection/sfd/sfd_detector.py` was
  read directly from GitHub; it calls
  `torch.utils.model_zoo.load_url('https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth')`.
  That exact URL was curl-checked in this session and returned `HTTP 200`.
  This is the canonical repo, not a fork — reports that it's "well known"
  to use a dead link were not reproduced here.
- **Both target machines have NVIDIA GPUs.** The desktop has one (used
  throughout this project); the friend's laptop has an RTX 3050
  (confirmed earlier in this conversation). Wav2Lip's CPU-mode slowness —
  a common complaint in community reports — is a fallback concern, not
  the primary path: torch should pick CUDA (`torch.cuda.is_available()`)
  on both machines.

## 3b. Two architectural decisions settled by this review

A fresh-context review of this document (see §3a) also surfaced two real
open questions that a wrong guess would have made expensive. Both are
settled here rather than left to the plan:

- **Vendored Wav2Lip runs as a subprocess, in its own virtual
  environment** — not imported in-process into the Flask app's existing
  `.venv`. Two independent reasons converge on this: (1) the anti-goal
  requires that a lip-sync failure can't crash the rest of the app;
  subprocess isolation gives that for free (an OOM or crash in the
  Wav2Lip process can't take down the Flask process), where an in-process
  `import` could. (2) Wav2Lip's own dependency versions (old
  numpy/opencv/librosa pins) have no reason to match what's already
  pinned in this project's `requirements.txt` for mediapipe/insightface —
  a second, isolated venv avoids any collision entirely, and lets each
  side use whatever versions actually work for it. The Flask route
  shells out (`subprocess.run`) to that second venv's Python with the
  vendored `inference.py`, waits for it to finish, then reads the output
  file it wrote.
- **`app.py`'s `app.run(...)` gains `threaded=True`.** A lip-sync request
  can take tens of seconds to a couple of minutes. Without this, Flask's
  dev server serializes requests — someone using the detect or swap tab
  while a lip-sync job is running would simply hang, which is exactly the
  kind of cross-tab breakage the anti-goal forbids. `threaded=True` lets
  the dev server handle those routes on a different thread while the
  lip-sync request's thread sits blocked on `subprocess.run` (which
  releases the GIL while waiting on the child process).

## 4. Constraints discovered

- **ffmpeg is now a required system dependency**, not just a pip package —
  Wav2Lip's own code shells out to the `ffmpeg` CLI for audio conversion
  and for muxing the final output. Must be installed and on `PATH` on both
  the desktop and the laptop; this is new relative to every earlier phase.
- **Wav2Lip's GitHub repo has no declared license** (no LICENSE file,
  GitHub reports `license: None`). Our project repo is **public**. To stay
  on clean legal ground, Wav2Lip's source files are used locally (cloned
  into a git-ignored folder, the same treatment already given to model
  weights) rather than copied into our public repository. Only our own
  glue code (the Flask route, the one-line librosa patch applied to the
  vendored copy, the watermark/ffmpeg wiring) is ours to commit.
- **Output is video, not a single image** — the existing pattern of
  returning a base64-encoded result inside a JSON response does not scale
  to video file sizes. See §5.
- Video processing is much slower than the existing single-image routes
  (expect tens of seconds, possibly minutes, depending on clip length and
  hardware) — the UI must set the expectation that this takes a while
  rather than imply something is broken.

## 5. Decisions already made (settled in conversation)

- **Input:** a video file (containing a face, someone talking) and a
  separate audio file. Output: that same video, mouth re-synced to the new
  audio.
- **Model:** Wav2Lip, using the GAN checkpoint (`wav2lip_gan.pth`, higher
  quality than the base checkpoint) from the HuggingFace mirror
  `Nekochu/Wav2Lip`.
- **Vendoring approach:** clone the canonical `Rudrabha/Wav2Lip` GitHub
  repo (verified working in §3a — not a fork) into a git-ignored local
  folder (e.g. `vendor/wav2lip/`, path TBD at plan time), pinned to a
  specific commit hash (captured at clone time) for reproducibility, not
  committed to this project's repo. It gets its own dedicated virtual
  environment per §3b, separate from the main Flask app's `.venv`.
  `models/README.md`-style setup instructions tell both machines how to
  set this up and where to place the downloaded checkpoints.
- **FR10 (real-time) is explicitly still required by the instructor**, but
  is its own later phase — not designed here.

## 6. Left to the builder (this document's author's calls — flag if wrong)

- **Output delivery:** save the result video to a server-side output
  directory (git-ignored, same treatment as `uploads/`/`outputs/` already
  in `.gitignore`) and serve it through a small dedicated route (e.g.
  `GET /lipsync/result/<token>.mp4`) rather than inlining it as base64 in
  JSON; the frontend's `/lipsync` POST response returns a URL, and the
  "after" panel uses an HTML `<video controls>` element pointed at that
  URL instead of an `<img>`. The "before" video is shown client-side from
  the uploaded file via `URL.createObjectURL`, same pattern already used
  for images.
- **Watermark on video:** apply the same "AI GENERATED" disclosure via
  ffmpeg's `drawtext` filter as part of the final muxing step (one extra
  ffmpeg filter argument), rather than decoding/re-encoding every frame in
  Python — much cheaper and reuses the same ffmpeg invocation Wav2Lip
  already does.
- **Size/length limits:** cap uploads (e.g. video ≤ 20 MB, audio ≤ 10 MB)
  to keep worst-case processing time bounded and predictable for a live
  demo. Since both machines have an NVIDIA GPU (§3a), the vendored
  Wav2Lip process should request CUDA when available and only fall back
  to CPU if it isn't — keeping actual processing time well under the
  CPU-only worst case the original research assumed. Exact size/length
  numbers are a plan-time detail, adjustable.
- **Processing UX:** just a status message along the lines of "İşleniyor...
  bu biraz uzun sürebilir" (like the existing tabs, no fake progress bar) — a real
  progress percentage isn't feasible without a bigger architecture change
  (polling/streaming), which is out of scope for this phase.
- **Metrics shown:** processing time only, in the same `readout`
  monospace-strip style as the other two tabs, for consistency.
- **Test fixtures:** a short local video (a few seconds, one person
  talking) and a short audio clip, under `tests/fixtures/`, git-ignored
  exactly like `sample_face.jpg` — tests that need them skip with a clear
  message when absent, same pattern as the existing fixture-gated tests.
- **UI tab:** "03 — Dudak Senkronu", added to the existing tab bar
  alongside "01 — Tespit" / "02 — Değiştirme", following the established
  module markup/CSS conventions exactly (same `.module-form`, `.readout`,
  before/after layout, just with `<video>` instead of `<img>` in the
  after/before panels).
