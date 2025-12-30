import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os
import sys
import time
from math import cos, sin
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(backend_dir)
from utils import normalize_landmarks

model = tf.keras.models.load_model(
    os.path.join(backend_dir, "head_pose_model.h5"),
    compile=False
)

model_path = os.path.join(backend_dir, "face_landmarker.task")
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)

def draw_axes(img, pitch, yaw, roll, tx, ty, size=50):
    # 3 angles are in radians
    p = pitch
    y = yaw
    r = roll

    # X-Axis (Pitch) - Red
    x1 = size * (cos(y) * cos(r)) + tx
    y1 = size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + ty

    # Y-Axis (Yaw) - Green
    x2 = size * (-cos(y) * sin(r)) + tx
    y2 = size * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + ty

    # Z-Axis (Roll) - Blue
    x3 = size * (sin(y)) + tx
    y3 = size * (-cos(y) * sin(p)) + ty

    # Draw the lines from the origin (tx, ty) to the endpoints
    cv2.line(img, (int(tx), int(ty)), (int(x1), int(y1)), (0, 0, 255), 3) # X - Red
    cv2.line(img, (int(tx), int(ty)), (int(x2), int(y2)), (0, 255, 0), 3) # Y - Green
    cv2.line(img, (int(tx), int(ty)), (int(x3), int(y3)), (255, 0, 0), 3) # Z - Blue

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()

    if not success:
        break

    image = cv2.flip(image, 1)

    # Process image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # MediaPipe Tasks requires an mp.Image object and a timestamp
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    frame_timestamp_ms = int(time.time() * 1000)
    
    # Perform the task
    results = detector.detect_for_video(mp_image, frame_timestamp_ms)

    if results.face_landmarks:
        for face_landmarks in results.face_landmarks:
            features = normalize_landmarks(face_landmarks) 
            
            # We add [np.newaxis] because the model expects a "batch" of images
            prediction = model.predict(features[np.newaxis], verbose=0)
            pitch, yaw, roll = prediction[0]

            h, w, _ = image.shape
            nose_tip = face_landmarks[4]
            nose_x, nose_y = int(nose_tip.x * w), int(nose_tip.y * h)

            # Draw the 3D axes on the nose
            draw_axes(image, pitch, -yaw, roll, nose_x, nose_y)

            # Display info
            cv2.putText(image, f"Pitch: {np.rad2deg(pitch):.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, f"Yaw: {np.rad2deg(yaw):.2f}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f"Roll: {np.rad2deg(roll):.2f}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Loomis Lens Live', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Clean up
detector.close()
cap.release()
cv2.destroyAllWindows()