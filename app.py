import base64
import pathlib
import re
import time
import uuid

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory

from face_detect import NoFaceDetectedError, detect_landmarks, estimate_head_pose
from face_swap import compute_identity_similarity, swap_faces
from lip_sync import NO_FACE_MESSAGE, LipSyncUnavailableError, run_lip_sync
from watermark import add_disclosure_watermark

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024  # 40 MB (lip-sync needs more than a single image)

UPLOADS_DIR = pathlib.Path("uploads")
OUTPUTS_DIR = pathlib.Path("outputs")


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/detect")
def detect():
    uploaded_file = request.files.get("image")
    if uploaded_file is None or uploaded_file.filename == "":
        return jsonify(error="No image uploaded."), 400

    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return jsonify(error="Could not decode image. Use PNG or JPEG."), 400

    start = time.perf_counter()
    try:
        landmarks, annotated_bgr = detect_landmarks(image_bgr)
    except NoFaceDetectedError as exc:
        return jsonify(error=str(exc)), 422

    yaw, pitch, roll = estimate_head_pose(landmarks, image_bgr.shape)
    elapsed = time.perf_counter() - start

    success, encoded = cv2.imencode(".png", annotated_bgr)
    if not success:
        return jsonify(error="Failed to encode result image."), 500

    return jsonify(
        landmark_count=len(landmarks),
        annotated_image_png_base64=base64.b64encode(encoded.tobytes()).decode("ascii"),
        head_pose={"yaw": yaw, "pitch": pitch, "roll": roll},
        processing_time_seconds=round(elapsed, 3),
    )


@app.post("/swap")
def swap():
    source_file = request.files.get("source")
    target_file = request.files.get("target")
    if (
        source_file is None or source_file.filename == ""
        or target_file is None or target_file.filename == ""
    ):
        return jsonify(error="Both a source and a target image are required."), 400

    source_bytes = np.frombuffer(source_file.read(), dtype=np.uint8)
    source_bgr = cv2.imdecode(source_bytes, cv2.IMREAD_COLOR)
    target_bytes = np.frombuffer(target_file.read(), dtype=np.uint8)
    target_bgr = cv2.imdecode(target_bytes, cv2.IMREAD_COLOR)

    if source_bgr is None or target_bgr is None:
        return jsonify(error="Could not decode one of the images. Use PNG or JPEG."), 400

    start = time.perf_counter()
    try:
        swapped_bgr = swap_faces(source_bgr, target_bgr)
    except NoFaceDetectedError as exc:
        return jsonify(error=str(exc)), 422
    elapsed = time.perf_counter() - start

    identity_similarity = compute_identity_similarity(source_bgr, swapped_bgr)
    watermarked_bgr = add_disclosure_watermark(swapped_bgr)

    success, encoded = cv2.imencode(".png", watermarked_bgr)
    if not success:
        return jsonify(error="Failed to encode result image."), 500

    return jsonify(
        swapped_image_png_base64=base64.b64encode(encoded.tobytes()).decode("ascii"),
        processing_time_seconds=round(elapsed, 3),
        identity_similarity=identity_similarity,
    )


@app.post("/lipsync")
def lipsync():
    video_file = request.files.get("video")
    audio_file = request.files.get("audio")
    if (
        video_file is None or video_file.filename == ""
        or audio_file is None or audio_file.filename == ""
    ):
        return jsonify(error="Both a video and an audio file are required."), 400

    UPLOADS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)

    token = uuid.uuid4().hex
    video_path = UPLOADS_DIR / f"{token}_video.mp4"
    audio_path = UPLOADS_DIR / f"{token}_audio.wav"
    output_path = OUTPUTS_DIR / f"{token}.mp4"
    video_file.save(video_path)
    audio_file.save(audio_path)

    try:
        start = time.perf_counter()
        try:
            run_lip_sync(video_path, audio_path, output_path)
        except LipSyncUnavailableError as exc:
            return jsonify(error=str(exc)), 503
        except RuntimeError as exc:
            app.logger.error("Lip sync failed: %s", exc)
            message = (
                NO_FACE_MESSAGE if str(exc) == NO_FACE_MESSAGE
                else "Lip sync processing failed for this video/audio pair."
            )
            return jsonify(error=message), 500
        elapsed = time.perf_counter() - start
    finally:
        video_path.unlink(missing_ok=True)
        audio_path.unlink(missing_ok=True)

    return jsonify(
        result_url=f"/lipsync/result/{token}.mp4",
        processing_time_seconds=round(elapsed, 3),
    )


@app.get("/lipsync/result/<filename>")
def lipsync_result(filename):
    if not re.fullmatch(r"[0-9a-f]{32}\.mp4", filename):
        return jsonify(error="Not found."), 404
    return send_from_directory(OUTPUTS_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
