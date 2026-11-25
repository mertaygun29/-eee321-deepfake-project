import pathlib

import pytest

from lip_sync import NO_FACE_MESSAGE, LipSyncUnavailableError, run_lip_sync

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


def test_run_lip_sync_gives_a_specific_message_when_no_face_found(tmp_path, monkeypatch):
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
        lambda *a, **k: subprocess.CompletedProcess(
            a, returncode=1, stdout="",
            stderr="ValueError: Face not detected! Ensure the video contains a face in all the frames.",
        ),
    )

    with pytest.raises(RuntimeError) as exc_info:
        run_lip_sync(tmp_path / "video.mp4", tmp_path / "audio.wav", tmp_path / "out.mp4")
    assert str(exc_info.value) == NO_FACE_MESSAGE


def test_run_lip_sync_raises_when_inference_produces_no_output(tmp_path, monkeypatch):
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
        lambda *a, **k: subprocess.CompletedProcess(a, returncode=0, stdout="", stderr=""),
    )

    with pytest.raises(RuntimeError):
        run_lip_sync(tmp_path / "video.mp4", tmp_path / "audio.wav", tmp_path / "out.mp4")


def test_run_lip_sync_raises_runtime_error_when_watermarking_fails(tmp_path, monkeypatch):
    import subprocess

    import lip_sync

    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"fake")
    fake_venv_python = tmp_path / "python.exe"
    fake_venv_python.write_bytes(b"fake")
    monkeypatch.setattr(lip_sync, "_CHECKPOINT_PATH", checkpoint)
    monkeypatch.setattr(lip_sync, "_VENV_PYTHON", fake_venv_python)

    output_path = tmp_path / "out.mp4"
    raw_output = output_path.with_name(output_path.stem + "_raw.mp4")
    call_count = {"value": 0}

    def fake_run(args, **kwargs):
        call_count["value"] += 1
        if call_count["value"] == 1:
            raw_output.write_bytes(b"fake raw video")
            return subprocess.CompletedProcess(args, returncode=0, stdout="", stderr="")
        raise subprocess.CalledProcessError(1, args, stderr=b"ffmpeg exploded")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError):
        run_lip_sync(tmp_path / "video.mp4", tmp_path / "audio.wav", output_path)
    assert not raw_output.exists()  # cleaned up even though the watermark step failed


def test_run_lip_sync_serializes_concurrent_calls(tmp_path, monkeypatch):
    import subprocess
    import threading
    import time

    import lip_sync

    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"fake")
    fake_venv_python = tmp_path / "python.exe"
    fake_venv_python.write_bytes(b"fake")
    monkeypatch.setattr(lip_sync, "_CHECKPOINT_PATH", checkpoint)
    monkeypatch.setattr(lip_sync, "_VENV_PYTHON", fake_venv_python)

    counter_lock = threading.Lock()
    active = {"value": 0}
    max_active = {"value": 0}

    def fake_run(*args, **kwargs):
        with counter_lock:
            active["value"] += 1
            max_active["value"] = max(max_active["value"], active["value"])
        time.sleep(0.05)
        with counter_lock:
            active["value"] -= 1
        return subprocess.CompletedProcess(args, returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)

    def call():
        try:
            run_lip_sync(tmp_path / "video.mp4", tmp_path / "audio.wav", tmp_path / "out.mp4")
        except RuntimeError:
            pass

    threads = [threading.Thread(target=call) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert max_active["value"] == 1


@REQUIRES_WAV2LIP
@REQUIRES_FIXTURES
def test_run_lip_sync_produces_watermarked_output(tmp_path):
    output_path = tmp_path / "result.mp4"
    run_lip_sync(FIXTURE_VIDEO, FIXTURE_AUDIO, output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0
