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

def load_loomis_mesh_with_faces(file_path):
    vertices = []
    faces = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith('f '):
                # OBJ indices start at 1, so we subtract 1 for Python
                face = [int(i.split('/')[0]) - 1 for i in line.split()[1:]]
                faces.append(face)
    return np.array(vertices), faces

mesh_path = os.path.join(backend_dir, "assets", "loomis_base.obj")
loomis_vertices, loomis_faces = load_loomis_mesh_with_faces(mesh_path)

def draw_solid_loomis_overlay(img, mesh_points, faces, pitch, yaw, roll, tx, ty, scale=100):
    R_x = np.array([[1, 0, 0], [0, cos(pitch), -sin(pitch)], [0, sin(pitch), cos(pitch)]])
    R_y = np.array([[cos(yaw), 0, sin(yaw)], [0, 1, 0], [-sin(yaw), 0, cos(yaw)]])
    R_z = np.array([[cos(roll), -sin(roll), 0], [sin(roll), cos(roll), 0], [0, 0, 1]])
    R = R_z @ R_y @ R_x

    projected = []
    for p in mesh_points:
        rotated_p = R @ np.array(p)
        x = int(rotated_p[0] * scale + tx)
        y = int(rotated_p[1] * scale + ty)
        projected.append([x, y])
    
    projected = np.array(projected)
    for face in faces:
        pts = projected[face].astype(np.int32)
        cv2.fillPoly(img, [pts], (200, 200, 200)) 
        cv2.polylines(img, [pts], True, (100, 100, 100), 1)
    return img

def draw_axes(img, pitch, yaw, roll, tx, ty, size=50):
    # 3 angles are in radians
    p = pitch
    y = yaw
    r = roll

    # X-Axis (Pitch) - Red
    x1 = size * (cos(y) * cos(r)) + tx
    y1 = size * (cos(p) * sin(r) + sin(p) * cos(r) * sin(y)) + ty

    # Y-Axis (Yaw) - Green
    x2 = size * (-cos(y) * sin(r)) + tx
    y2 = size * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + ty

    # Z-Axis (Roll) - Blue
    x3 = size * (sin(y)) + tx
    y3 = size * (-sin(p) * cos(y)) + ty

    # Draw the lines from the origin (tx, ty) to the endpoints
    cv2.line(img, (int(tx), int(ty)), (int(x1), int(y1)), (0, 0, 255), 3) # X-Axis: Red
    cv2.line(img, (int(tx), int(ty)), (int(x2), int(y2)), (0, 255, 0), 3) # Y-Axis: Green
    cv2.line(img, (int(tx), int(ty)), (int(x3), int(y3)), (255, 0, 0), 3) # Z-Axis: Blue

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
        # Create a copy for the semi-transparent overlay
        overlay = image.copy()
        h, w, _ = image.shape

        for face_landmarks in results.face_landmarks:
            # Draw MediaPipe landmarks
            for idx, landmark in enumerate(face_landmarks):
                lx = int(landmark.x * w)
                ly = int(landmark.y * h)
                cv2.circle(image, (lx, ly), 1, (255, 255, 255), -1) 
            
            features = normalize_landmarks(face_landmarks) 
            
            # Prediction expects a batch
            prediction = model.predict(features[np.newaxis], verbose=0)
            pitch, yaw, roll = prediction[0]
            current_yaw = -yaw

            nose_tip = face_landmarks[4]
            nose_x, nose_y = int(nose_tip.x * w), int(nose_tip.y * h)

            # Draw the 3D axes on the nose
            draw_axes(image, pitch, current_yaw, roll, nose_x, nose_y)

            # Ear-to-ear landmarks for face width and centering
            ear_r = np.array([face_landmarks[234].x * w, face_landmarks[234].y * h])
            ear_l = np.array([face_landmarks[454].x * w, face_landmarks[454].y * h])
            
            # Center of the line between ears
            mid_ears_x = (ear_r[0] + ear_l[0]) / 2
            # Vertical height anchor from landmark 8 (mid-brow)
            brow_y = face_landmarks[8].y * h

            # Match mesh scale to the ear-to-ear width
            face_width = np.linalg.norm(ear_r - ear_l)
            dynamic_scale = face_width / 2.0 

            # Offset anchor into the skull using yaw and pitch
            z_depth = dynamic_scale * 0.9
            center_x = mid_ears_x + (z_depth * sin(current_yaw))
            center_y = brow_y + (z_depth * sin(pitch))

            # Draw Loomis Mesh on the overlay copy
            overlay = draw_solid_loomis_overlay(overlay, loomis_vertices, loomis_faces, 
                                               pitch, current_yaw, roll, center_x, center_y, 
                                               scale=dynamic_scale)

            # Display info
            cv2.putText(image, f"Pitch: {np.rad2deg(pitch):.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, f"Yaw: {np.rad2deg(current_yaw):.2f}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f"Roll: {np.rad2deg(roll):.2f}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Merge the solid overlay with the base image
        image = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)
    
    cv2.imshow('Loomis Lens Live', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

detector.close()
cap.release()
cv2.destroyAllWindows()