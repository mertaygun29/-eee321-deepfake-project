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
