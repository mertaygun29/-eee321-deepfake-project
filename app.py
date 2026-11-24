import base64

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from face_detect import NoFaceDetectedError, detect_landmarks

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

    success, encoded = cv2.imencode(".png", annotated_bgr)
    if not success:
        return jsonify(error="Failed to encode result image."), 500

    return jsonify(
        landmark_count=len(landmarks),
        annotated_image_png_base64=base64.b64encode(encoded.tobytes()).decode("ascii"),
    )


if __name__ == "__main__":
    app.run(debug=True)
