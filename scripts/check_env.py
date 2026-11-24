import cv2
import flask
import mediapipe as mp
import numpy as np

print("flask", flask.__version__)
print("opencv-python", cv2.__version__)
print("mediapipe", mp.__version__)
print("numpy", np.__version__)

# mediapipe 1.x dropped the legacy mp.solutions API; face landmark detection
# now lives under mp.tasks.vision.FaceLandmarker and needs a downloaded
# .task model file, which Task 2 handles. Here we just confirm the module
# is importable and has the expected shape.
assert hasattr(mp.tasks.vision, "FaceLandmarker"), "mediapipe Tasks API not found"
print("mediapipe.tasks.vision.FaceLandmarker available OK")
