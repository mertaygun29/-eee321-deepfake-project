"""One-time setup for the lip-sync feature (Wav2Lip).

Run once per machine: `python scripts/setup_lipsync.py`
Safe to re-run — every step is idempotent (skips what's already done).

Needs `git` and `ffmpeg` on PATH; this script does not install those.
"""
import pathlib
import subprocess
import sys
import urllib.request

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
VENDOR_DIR = PROJECT_ROOT / "vendor" / "wav2lip"
REPO_DIR = VENDOR_DIR / "Wav2Lip"
VENV_DIR = VENDOR_DIR / ".venv"
VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"
PINNED_COMMIT = "bac9a81e63ecc153202353372e5724b83d9e6322"
CHECKPOINT_PATH = REPO_DIR / "checkpoints" / "wav2lip_gan.pth"
CHECKPOINT_URL = "https://huggingface.co/Nekochu/Wav2Lip/resolve/main/wav2lip_gan.pth"


def clone_repo() -> None:
    if REPO_DIR.exists():
        print(f"{REPO_DIR} already exists, skipping clone.")
        return
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "https://github.com/Rudrabha/Wav2Lip.git", str(REPO_DIR)],
        check=True,
    )
    subprocess.run(["git", "checkout", PINNED_COMMIT], cwd=str(REPO_DIR), check=True)


def apply_patches() -> None:
    _patch_file(
        REPO_DIR / "audio.py",
        "librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels,",
        "librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels,",
    )
    _patch_file(
        REPO_DIR / "face_detection" / "utils.py",
        "dtype=np.int)",
        "dtype=np.int32)",
    )


def _patch_file(path: pathlib.Path, old: str, new: str) -> None:
    content = path.read_text()
    if old not in content:
        print(f"{path.name} already patched, skipping.")
        return
    path.write_text(content.replace(old, new))
    print(f"Patched {path}.")


def create_venv() -> None:
    if VENV_PYTHON.exists():
        print(f"{VENV_DIR} already exists, skipping venv creation.")
        return
    subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)


def install_dependencies() -> None:
    subprocess.run(
        [
            str(VENV_PYTHON), "-m", "pip", "install",
            "torch==2.10.0+cu128", "--index-url", "https://download.pytorch.org/whl/cu128",
        ],
        check=True,
    )
    subprocess.run(
        [
            str(VENV_PYTHON), "-m", "pip", "install",
            "librosa==1.0.1", "numpy==2.5.5", "opencv-contrib-python==5.1.0.95",
            "tqdm==4.70.1", "scipy==1.18.1",
        ],
        check=True,
    )


def download_checkpoint() -> None:
    if CHECKPOINT_PATH.exists():
        print(f"{CHECKPOINT_PATH} already present, skipping.")
        return
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading wav2lip_gan.pth (~415 MB) ...")
    urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT_PATH)
    print(f"Saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    clone_repo()
    apply_patches()
    create_venv()
    install_dependencies()
    download_checkpoint()
    print("Lip-sync setup complete.")
