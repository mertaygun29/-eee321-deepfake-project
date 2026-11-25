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


NO_FACE_MESSAGE = (
    "No face detected in the video. Make sure a face is visible in every frame."
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
        if "Face not detected" in result.stderr:
            raise RuntimeError(NO_FACE_MESSAGE)
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
