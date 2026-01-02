import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os
import sys
import time
from math import cos, sin, sqrt
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
detector = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1
    )
)

def load_loomis_mesh_with_faces(file_path):
    vertices, faces = [], []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '): vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith('f '): faces.append([int(i.split('/')[0]) - 1 for i in line.split()[1:]])
    return np.array(vertices), faces

mesh_path = os.path.join(backend_dir, "assets", "loomis_base.obj")
loomis_vertices, loomis_faces = load_loomis_mesh_with_faces(mesh_path)

def draw_brow_axes(img, R, origin_2d, size=50):
    """Draws 3D axes (X-red, Y-green, Z-blue) starting from the brow landmark."""
    x_axis = np.array([1, 0, 0]) * size
    y_axis = np.array([0, 1, 0]) * size
    z_axis = np.array([0, 0, 1]) * size

    x_rot = R @ x_axis
    y_rot = R @ y_axis
    z_rot = R @ z_axis

    ox, oy = int(origin_2d[0]), int(origin_2d[1])
    cv2.line(img, (ox, oy), (int(ox + x_rot[0]), int(oy + x_rot[1])), (0, 0, 255), 2)
    cv2.line(img, (ox, oy), (int(ox + y_rot[0]), int(oy + y_rot[1])), (0, 255, 0), 2)
    cv2.line(img, (ox, oy), (int(ox + z_rot[0]), int(oy + z_rot[1])), (255, 0, 0), 2)

def draw_solid_loomis_overlay(img, mesh_points, faces, pitch, yaw, roll, tx, ty, scale=100, chin_target_2d=None, brow_target_2d=None):
    SIDE_DIST = 0.7454 
    SIDE_RADIUS = sqrt(1.0 - SIDE_DIST**2)

    R_x = np.array([[1, 0, 0], [0, cos(pitch), -sin(pitch)], [0, sin(pitch), cos(pitch)]])
    R_y = np.array([[cos(yaw), 0, sin(yaw)], [0, 1, 0], [-sin(yaw), 0, cos(yaw)]])
    R_z = np.array([[cos(roll), -sin(roll), 0], [sin(roll), cos(roll), 0], [0, 0, 1]])
    R = R_z @ R_y @ R_x

    transformed = []
    for p in mesh_points:
        rotated_p = R @ np.array(p)
        sx, sy = int(rotated_p[0] * scale + tx), int(rotated_p[1] * scale + ty)
        transformed.append([sx, sy, rotated_p[2]])
    transformed = np.array(transformed)

    face_depths = sorted([(i, np.mean(transformed[face, 2])) for i, face in enumerate(faces)], key=lambda x: x[1])

    temp_img = img.copy()

    for face_idx, face_z in face_depths:
        pts = transformed[faces[face_idx], :2].astype(np.int32)
        overlay_face = temp_img.copy()
        cv2.fillPoly(overlay_face, [pts], (240, 240, 240)) 
        cv2.polylines(overlay_face, [pts], True, (180, 180, 180), 1)
        cv2.addWeighted(overlay_face, 0.4, temp_img, 0.6, 0, temp_img)

    num_segments = 64
    angles = np.linspace(0, 2 * np.pi, num_segments)
    brow_pts = np.array([[cos(a), 0, sin(a)] for a in angles])
    median_pts = np.array([[0, cos(a), sin(a)] for a in angles])
    ear_line_pts = np.array([[cos(a), sin(a), 0] for a in angles])
    side_l_pts = np.array([[-SIDE_DIST, cos(a) * SIDE_RADIUS, sin(a) * SIDE_RADIUS] for a in angles])
    side_r_pts = np.array([[SIDE_DIST, cos(a) * SIDE_RADIUS, sin(a) * SIDE_RADIUS] for a in angles])

    def draw_line_3d(line_pts, color):
        for i in range(len(line_pts) - 1):
            p1, p2 = R @ line_pts[i], R @ line_pts[i+1]
            if p1[2] > -0.2:
                cv2.line(temp_img, (int(p1[0]*scale+tx), int(p1[1]*scale+ty)), 
                         (int(p2[0]*scale+tx), int(p2[1]*scale+ty)), color, 2)

    draw_line_3d(brow_pts, (255, 255, 0))   
    draw_line_3d(median_pts, (255, 0, 255)) 
    draw_line_3d(side_l_pts, (0, 255, 255)) 
    draw_line_3d(side_r_pts, (0, 255, 255)) 
    draw_line_3d(ear_line_pts, (0, 255, 255)) 

    # --- DRAW CHIN LINE ONLY ---
    if chin_target_2d is not None and brow_target_2d is not None:
        # Draw straight median line from Brow to Chin
        cv2.line(temp_img, brow_target_2d, chin_target_2d, (255, 255, 255), 2)

    cv2.circle(temp_img, (int(tx), int(ty)), 5, (0, 255, 0), -1)
    return temp_img

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success: break
    image = cv2.flip(image, 1)
    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    timestamp = int(time.time() * 1000)
    results = detector.detect_for_video(mp_image, timestamp)

    if results.face_landmarks:
        overlay_layer = image.copy()
        for face_landmarks in results.face_landmarks:
            brow_target = face_landmarks[9]   
            hairline = face_landmarks[10]     
            nose_base = face_landmarks[2]     
            chin_landmark = face_landmarks[152] 

            features = normalize_landmarks(face_landmarks) 
            prediction = model.predict(features[np.newaxis], verbose=0)
            pitch, yaw, roll = prediction[0]
            current_yaw = -yaw

            dist_vertical = np.linalg.norm(np.array([hairline.x*w, hairline.y*h]) - np.array([nose_base.x*w, nose_base.y*h]))
            dynamic_scale = (dist_vertical / (2 * sqrt(1.0 - 0.7454**2))) * 1.15

            R_x = np.array([[1, 0, 0], [0, cos(pitch), -sin(pitch)], [0, sin(pitch), cos(pitch)]])
            R_y = np.array([[cos(current_yaw), 0, sin(current_yaw)], [0, 1, 0], [-sin(current_yaw), 0, cos(current_yaw)]])
            R_z = np.array([[cos(roll), -sin(roll), 0], [sin(roll), cos(roll), 0], [0, 0, 1]])
            R_full = R_z @ R_y @ R_x

            surface_vector = R_full @ np.array([0, 0, 1.0])
            
            push_depth = 1.1
            tx = (brow_target.x * w) - (surface_vector[0] * dynamic_scale * push_depth)
            ty = (brow_target.y * h) - (surface_vector[1] * dynamic_scale * push_depth)

            brow_2d = (int(brow_target.x * w), int(brow_target.y * h))
            chin_2d = (int(chin_landmark.x * w), int(chin_landmark.y * h))

            overlay_layer = draw_solid_loomis_overlay(overlay_layer, loomis_vertices, loomis_faces, 
                                               pitch, current_yaw, roll, 
                                               tx, ty, scale=dynamic_scale,
                                               chin_target_2d=chin_2d,
                                               brow_target_2d=brow_2d)

            draw_brow_axes(overlay_layer, R_full, brow_2d, size=dynamic_scale * 0.5)

        image = cv2.addWeighted(overlay_layer, 0.6, image, 0.4, 0)
    
    cv2.imshow('Loomis Lens - Brow to Chin', image)
    if cv2.waitKey(5) & 0xFF == 27: break

detector.close()
cap.release()
cv2.destroyAllWindows()