import io
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
