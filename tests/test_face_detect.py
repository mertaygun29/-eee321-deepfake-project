import pathlib

import numpy as np
import pytest

from face_detect import NoFaceDetectedError, detect_landmarks

FIXTURE_PATH = pathlib.Path(__file__).parent / "fixtures" / "sample_face.jpg"
REQUIRES_FIXTURE = pytest.mark.skipif(
    not FIXTURE_PATH.exists(),
    reason=(
        "Add a real photo with a visible face at "
        "tests/fixtures/sample_face.jpg to run this test "
        "(file is gitignored, add it locally)."
    ),
)


def test_detect_landmarks_raises_on_blank_image():
    blank = np.zeros((200, 200, 3), dtype="uint8")
    with pytest.raises(NoFaceDetectedError):
        detect_landmarks(blank)


@REQUIRES_FIXTURE
def test_detect_landmarks_finds_face():
    import cv2

    image = cv2.imread(str(FIXTURE_PATH))
    landmarks, annotated = detect_landmarks(image)
    assert len(landmarks) == 478  # 468 mesh points + 10 iris points
    assert annotated.shape == image.shape


@REQUIRES_FIXTURE
def test_estimate_head_pose_returns_three_angles():
    import cv2

    from face_detect import estimate_head_pose

    image = cv2.imread(str(FIXTURE_PATH))
    landmarks, _ = detect_landmarks(image)
    yaw, pitch, roll = estimate_head_pose(landmarks, image.shape)
    for angle in (yaw, pitch, roll):
        assert isinstance(angle, float)
        assert -180.0 <= angle <= 180.0
