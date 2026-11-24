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


_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -330.0, -65.0),     # Chin
        (-225.0, 170.0, -135.0),  # Left eye, left corner
        (225.0, 170.0, -135.0),   # Right eye, right corner
        (-150.0, -150.0, -125.0),  # Left mouth corner
        (150.0, -150.0, -125.0),   # Right mouth corner
    ],
    dtype="double",
)

# Indices into the 478-point FaceMesh landmark list for the points above.
_POSE_LANDMARK_INDICES = (1, 152, 33, 263, 61, 291)


def estimate_head_pose(
    landmarks: list[tuple[float, float]], image_shape: tuple[int, int, int]
) -> tuple[float, float, float]:
    """Estimate (yaw, pitch, roll) in degrees from face-mesh landmarks."""
    height, width = image_shape[:2]
    image_points = np.array(
        [landmarks[index] for index in _POSE_LANDMARK_INDICES], dtype="double"
    )

    focal_length = width
    center = (width / 2, height / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype="double",
    )
    dist_coeffs = np.zeros((4, 1))

    _, rotation_vector, _ = cv2.solvePnP(
        _MODEL_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    pitch, yaw, roll = _rotation_matrix_to_euler_angles(rotation_matrix)

    # This model/axis convention reports a frontal face's pitch near +-180
    # degrees instead of 0; wrap it back into a sane +-90 range.
    if pitch > 90:
        pitch -= 180
    elif pitch < -90:
        pitch += 180

    return float(yaw), float(pitch), float(roll)


def _rotation_matrix_to_euler_angles(
    rotation_matrix: np.ndarray,
) -> tuple[float, float, float]:
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = 0.0

    return float(np.degrees(x)), float(np.degrees(y)), float(np.degrees(z))
