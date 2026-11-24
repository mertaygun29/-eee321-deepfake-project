import pathlib

import cv2
import mediapipe as mp
import numpy as np

_MODEL_PATH = pathlib.Path(__file__).parent / "models" / "weights" / "face_landmarker.task"
_landmarker = None


class NoFaceDetectedError(Exception):
    """Raised when no face can be found in the given image."""


def _get_landmarker():
    global _landmarker
    if _landmarker is None:
        if not _MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Missing model file: {_MODEL_PATH}. "
                "Run `python scripts/download_models.py` first."
            )
        base_options = mp.tasks.BaseOptions(model_asset_path=str(_MODEL_PATH))
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
        )
        _landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
    return _landmarker


def detect_landmarks(
    image_bgr: np.ndarray,
) -> tuple[list[tuple[float, float]], np.ndarray]:
    """Detect facial landmarks in a BGR image.

    Returns (landmarks, annotated_image_bgr): landmarks is a list of
    (x, y) pixel coordinates for every detected face-mesh point, and
    annotated_image_bgr is a copy of the input with landmark contours
    drawn on it. Raises NoFaceDetectedError if no face is found.
    """
    height, width = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    result = _get_landmarker().detect(mp_image)

    if not result.face_landmarks:
        raise NoFaceDetectedError("No face detected in the uploaded image.")

    face_landmarks = result.face_landmarks[0]
    landmarks = [(lm.x * width, lm.y * height) for lm in face_landmarks]

    annotated = image_bgr.copy()
    mp.tasks.vision.drawing_utils.draw_landmarks(
        image=annotated,
        landmark_list=face_landmarks,
        connections=mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.tasks.vision.drawing_styles.get_default_face_mesh_contours_style(),
    )
    return landmarks, annotated
