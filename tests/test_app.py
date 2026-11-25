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
