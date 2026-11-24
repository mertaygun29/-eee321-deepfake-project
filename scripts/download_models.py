"""Download pretrained model weights needed to run this project.

Run once per machine: `python scripts/download_models.py`
Weights are saved under models/weights/ and are git-ignored.
"""
import pathlib
import urllib.request

WEIGHTS_DIR = pathlib.Path(__file__).parent.parent / "models" / "weights"

MODELS = {
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    ),
    # Official InsightFace releases no longer host this file directly;
    # this is the community mirror used by most face-swap projects (roop,
    # ReActor, etc.) for an unauthenticated direct download.
    "inswapper_128.onnx": (
        "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/"
        "inswapper_128.onnx"
    ),
}


def download_models() -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in MODELS.items():
        destination = WEIGHTS_DIR / filename
        if destination.exists():
            print(f"{filename} already present, skipping.")
            continue
        print(f"Downloading {filename} from {url} ...")
        urllib.request.urlretrieve(url, destination)
        print(f"Saved to {destination}")


if __name__ == "__main__":
    download_models()
