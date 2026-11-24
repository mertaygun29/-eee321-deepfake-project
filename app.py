import base64

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from face_detect import NoFaceDetectedError, detect_landmarks, estimate_head_pose
from face_swap import swap_faces
from watermark import add_disclosure_watermark

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit


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

    try:
        landmarks, annotated_bgr = detect_landmarks(image_bgr)
    except NoFaceDetectedError as exc:
        return jsonify(error=str(exc)), 422

    yaw, pitch, roll = estimate_head_pose(landmarks, image_bgr.shape)

    success, encoded = cv2.imencode(".png", annotated_bgr)
    if not success:
        return jsonify(error="Failed to encode result image."), 500

    return jsonify(
        landmark_count=len(landmarks),
        annotated_image_png_base64=base64.b64encode(encoded.tobytes()).decode("ascii"),
        head_pose={"yaw": yaw, "pitch": pitch, "roll": roll},
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

    try:
        swapped_bgr = swap_faces(source_bgr, target_bgr)
    except NoFaceDetectedError as exc:
        return jsonify(error=str(exc)), 422

    watermarked_bgr = add_disclosure_watermark(swapped_bgr)

    success, encoded = cv2.imencode(".png", watermarked_bgr)
    if not success:
        return jsonify(error="Failed to encode result image."), 500

    return jsonify(
        swapped_image_png_base64=base64.b64encode(encoded.tobytes()).decode("ascii"),
    )


if __name__ == "__main__":
    app.run(debug=True)
