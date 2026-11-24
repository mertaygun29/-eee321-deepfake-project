import pathlib

import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

from face_detect import NoFaceDetectedError

_SWAPPER_MODEL_PATH = (
    pathlib.Path(__file__).parent / "models" / "weights" / "inswapper_128.onnx"
)

_analyzer = None
_swapper = None


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        _analyzer = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        _analyzer.prepare(ctx_id=0, det_size=(640, 640))
    return _analyzer


def _get_swapper():
    global _swapper
    if _swapper is None:
        if not _SWAPPER_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Missing model file: {_SWAPPER_MODEL_PATH}. "
                "Run `python scripts/download_models.py` first."
            )
        _swapper = get_model(str(_SWAPPER_MODEL_PATH), providers=["CPUExecutionProvider"])
    return _swapper


def swap_faces(source_bgr: np.ndarray, target_bgr: np.ndarray) -> np.ndarray:
    """Swap the face from source_bgr onto the face found in target_bgr.

    Returns the swapped image (same shape as target_bgr). Raises
    NoFaceDetectedError if either image has no detectable face.
    """
    analyzer = _get_analyzer()

    source_faces = analyzer.get(source_bgr)
    if not source_faces:
        raise NoFaceDetectedError("No face detected in the source image.")

    target_faces = analyzer.get(target_bgr)
    if not target_faces:
        raise NoFaceDetectedError("No face detected in the target image.")

    return _get_swapper().get(
        target_bgr, target_faces[0], source_faces[0], paste_back=True
    )
