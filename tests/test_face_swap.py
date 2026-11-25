import pathlib

import cv2
import numpy as np
import pytest

import face_swap
from face_detect import NoFaceDetectedError
from face_swap import compute_identity_similarity, swap_faces

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


@REQUIRES_FIXTURE
def test_compute_identity_similarity_is_high_for_self_swap():
    image = cv2.imread(str(FIXTURE_PATH))
    result = swap_faces(image, image)
    similarity = compute_identity_similarity(image, result)
    assert similarity is not None
    assert similarity > 0.5


@REQUIRES_FIXTURE
def test_compute_identity_similarity_returns_none_when_result_has_no_face():
    image = cv2.imread(str(FIXTURE_PATH))
    blank = np.zeros((200, 200, 3), dtype="uint8")
    assert compute_identity_similarity(image, blank) is None


class _FakeFace:
    def __init__(self, embedding):
        self.normed_embedding = embedding


class _FakeAnalyzer:
    def __init__(self, source_faces, result_faces):
        self._source_faces = source_faces
        self._result_faces = result_faces
        self._calls = 0

    def get(self, image):
        self._calls += 1
        return self._source_faces if self._calls == 1 else self._result_faces


def test_compute_identity_similarity_picks_best_matching_face_in_multi_face_result(monkeypatch):
    source_embedding = np.array([1.0, 0.0])
    bystander_embedding = np.array([0.0, 1.0])  # orthogonal to source -> similarity 0
    swapped_embedding = np.array([1.0, 0.0])  # identical to source -> similarity 1

    fake_analyzer = _FakeAnalyzer(
        source_faces=[_FakeFace(source_embedding)],
        # Swapped face intentionally placed second, as detection confidence
        # ordering is not guaranteed to put it first.
        result_faces=[_FakeFace(bystander_embedding), _FakeFace(swapped_embedding)],
    )
    monkeypatch.setattr(face_swap, "_get_analyzer", lambda: fake_analyzer)

    blank = np.zeros((10, 10, 3), dtype="uint8")
    similarity = compute_identity_similarity(blank, blank)
    assert similarity == pytest.approx(1.0)
