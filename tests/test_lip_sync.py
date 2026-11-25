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
