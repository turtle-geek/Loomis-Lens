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
                face = [int(i.split('/')[0]) - 1 for i in line.split()[1:]]
                faces.append(face)
    return np.array(vertices), faces

mesh_path = os.path.join(backend_dir, "assets", "loomis_base.obj")
loomis_vertices, loomis_faces = load_loomis_mesh_with_faces(mesh_path)

def draw_solid_loomis_overlay(img, mesh_points, faces, pitch, yaw, roll, tx, ty, scale=100):
    # Rotation Matrices
    R_x = np.array([[1, 0, 0], [0, cos(pitch), -sin(pitch)], [0, sin(pitch), cos(pitch)]])
    R_y = np.array([[cos(yaw), 0, sin(yaw)], [0, 1, 0], [-sin(yaw), 0, cos(yaw)]])
    R_z = np.array([[cos(roll), -sin(roll), 0], [sin(roll), cos(roll), 0], [0, 0, 1]])
    R = R_z @ R_y @ R_x

    # Transform all vertices and store rotated Z for depth sorting
    transformed_points = []
    for p in mesh_points:
        rotated_p = R @ np.array(p)
        # Screen projection
        sx = int(rotated_p[0] * scale + tx)
        sy = int(rotated_p[1] * scale + ty)
        # Keep rotated_p[2] as the depth (Z)
        transformed_points.append([sx, sy, rotated_p[2]])
    
    transformed_points = np.array(transformed_points)

    # Sort faces by average depth so back faces are drawn first
    face_depths = []
    for i, face in enumerate(faces):
        avg_z = np.mean(transformed_points[face, 2])
        face_depths.append((i, avg_z))
    
    # Sort faces from furthest (lowest Z) to closest (highest Z)
    face_depths.sort(key=lambda x: x[1])

    # Draw the sorted faces
    for face_idx, _ in face_depths:
        face = faces[face_idx]
        # Get only X,Y for fillPoly
        pts = transformed_points[face, :2].astype(np.int32)
        
        # Solid fill (Light gray)
        cv2.fillPoly(img, [pts], (220, 220, 220)) 
        # Clean outlines (Darker gray)
        cv2.polylines(img, [pts], True, (120, 120, 120), 1)
    
    return img

def draw_axes(img, pitch, yaw, roll, tx, ty, size=50):
    p, y, r = pitch, yaw, roll
    x1 = size * (cos(y) * cos(r)) + tx
    y1 = size * (cos(p) * sin(r) + sin(p) * cos(r) * sin(y)) + ty
    x2 = size * (-cos(y) * sin(r)) + tx
    y2 = size * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + ty
    x3 = size * (sin(y)) + tx
    y3 = size * (-sin(p) * cos(y)) + ty
    cv2.line(img, (int(tx), int(ty)), (int(x1), int(y1)), (0, 0, 255), 3)
    cv2.line(img, (int(tx), int(ty)), (int(x2), int(y2)), (0, 255, 0), 3)
    cv2.line(img, (int(tx), int(ty)), (int(x3), int(y3)), (255, 0, 0), 3)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    frame_timestamp_ms = int(time.time() * 1000)
    results = detector.detect_for_video(mp_image, frame_timestamp_ms)

    if results.face_landmarks:
        overlay = image.copy()
        h, w, _ = image.shape

        for face_landmarks in results.face_landmarks:
            for idx, landmark in enumerate(face_landmarks):
                lx, ly = int(landmark.x * w), int(landmark.y * h)
                # cv2.circle(image, (lx, ly), 1, (255, 255, 255), -1) 
            
            features = normalize_landmarks(face_landmarks) 
            prediction = model.predict(features[np.newaxis], verbose=0)
            pitch, yaw, roll = prediction[0]
            current_yaw = -yaw

            nose_tip = face_landmarks[4]
            nose_x, nose_y = int(nose_tip.x * w), int(nose_tip.y * h)
            draw_axes(image, pitch, current_yaw, roll, nose_x, nose_y)

            ear_r = np.array([face_landmarks[234].x * w, face_landmarks[234].y * h])
            ear_l = np.array([face_landmarks[454].x * w, face_landmarks[454].y * h])
            mid_ears_x = (ear_r[0] + ear_l[0]) / 2
            brow_y = face_landmarks[8].y * h

            face_width = np.linalg.norm(ear_r - ear_l)
            dynamic_scale = (face_width / 2.0) * 1.35

            z_depth_factor = dynamic_scale * 0.55

            # Anchor point values for the loomis base
            center_x = mid_ears_x + (z_depth_factor * sin(current_yaw) * cos(pitch))
            center_y = brow_y + (z_depth_factor * sin(pitch) * cos(current_yaw))

            overlay = draw_solid_loomis_overlay(overlay, loomis_vertices, loomis_faces, 
                                               pitch, current_yaw, roll, center_x, center_y, 
                                               scale=dynamic_scale)

            cv2.putText(image, f"Pitch: {np.rad2deg(pitch):.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, f"Yaw: {np.rad2deg(current_yaw):.2f}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f"Roll: {np.rad2deg(roll):.2f}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        image = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)
    
    cv2.imshow('Loomis Lens Live', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

detector.close()
cap.release()
cv2.destroyAllWindows()