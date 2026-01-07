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

# Config - Aligned with App.py
RENDER_DAMPING = 0.9  
alpha_angles = 0.85
alpha_center = 0.75
JAW_L_IDX = [58, 172, 136, 150, 149, 176, 148, 152]
JAW_R_IDX = [288, 397, 365, 379, 378, 400, 377, 152]

# Path setup
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from src.utils import normalize_landmarks

# Model initialization
model = tf.keras.models.load_model(
    os.path.join(backend_dir, "models", "head_pose_model.h5"),
    compile=False
)

model_path = os.path.join(backend_dir, "models", "face_landmarker.task")
detector = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1
    )
)

def load_loomis_mesh(file_path):
    vertices, faces = [], []
    if not os.path.exists(file_path):
        return np.array([]), []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '): vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith('f '): faces.append([int(i.split('/')[0]) - 1 for i in line.split()[1:]])
    return np.array(vertices), faces

mesh_path = os.path.join(backend_dir, "assets", "loomis_base.obj")
loomis_vertices, loomis_faces = load_loomis_mesh(mesh_path)

def draw_loomis_overlay(img, mesh_points, faces, pitch, yaw, roll, tx, ty, scale=100, 
                        nose_2d=None, chin_2d=None, jaw_pts_l=None, jaw_pts_r=None):
    SIDE_DIST = 0.7454 
    SIDE_RADIUS = sqrt(1.0 - SIDE_DIST**2)
    blue_color = (255, 120, 0)
    red_color = (0, 0, 255) 
    green_color = (0, 255, 0)
    yellow_color = (0, 255, 255) 
    thickness = 2

    dp, dy, dr = pitch * RENDER_DAMPING, yaw * RENDER_DAMPING, roll * RENDER_DAMPING
    R_x = np.array([[1, 0, 0], [0, cos(dp), -sin(dp)], [0, sin(dp), cos(dp)]])
    R_y = np.array([[cos(dy), 0, sin(dy)], [0, 1, 0], [-sin(dy), 0, cos(dy)]])
    R_z = np.array([[cos(dr), -sin(dr), 0], [sin(dr), cos(dr), 0], [0, 0, 1]])
    R = R_z @ R_y @ R_x

    overlay = np.zeros_like(img)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    if len(mesh_points) > 0:
        transformed = []
        for p in mesh_points:
            rotated_p = R @ np.array(p)
            sx, sy = int(rotated_p[0] * scale + tx), int(rotated_p[1] * scale + ty)
            transformed.append([sx, sy, rotated_p[2]])
        transformed = np.array(transformed)

        face_depths = sorted([(i, np.mean(transformed[face, 2])) for i, face in enumerate(faces)], key=lambda x: x[1])
        for face_idx, _ in face_depths:
            pts = transformed[faces[face_idx], :2].astype(np.int32)
            cv2.fillPoly(overlay, [pts], (242, 242, 242)) 
            cv2.fillPoly(mask, [pts], 255)

    def draw_3d_segment(target_surface, p1_3d, p2_3d, color, vibrant=False):
        r1, r2 = R @ p1_3d, R @ p2_3d
        is_front = (r1[2] > -0.1 or r2[2] > -0.1)
        if vibrant != is_front: return
        pt1 = (int(r1[0] * scale + tx), int(r1[1] * scale + ty))
        pt2 = (int(r2[0] * scale + tx), int(r2[1] * scale + ty))
        cv2.line(target_surface, pt1, pt2, color, thickness)

    angles = np.linspace(0, 2 * np.pi, 120)
    brow_ring = np.array([[cos(a), 0, sin(a)] for a in angles])
    median_ring = np.array([[0, cos(a), sin(a)] for a in angles])
    
    hinge_l_3d = np.array([-SIDE_DIST, SIDE_RADIUS * 0.5, 0.0])
    corner_l_3d = np.array([-SIDE_DIST, SIDE_RADIUS * 0.9, 0.1])
    hinge_r_3d = np.array([SIDE_DIST, SIDE_RADIUS * 0.5, 0.0])
    corner_r_3d = np.array([SIDE_DIST, SIDE_RADIUS * 0.9, 0.1])
    
    for i in range(len(brow_ring)-1):
        if abs(brow_ring[i][0]) <= SIDE_DIST:
            draw_3d_segment(overlay, brow_ring[i], brow_ring[i+1], blue_color, vibrant=False)
        draw_3d_segment(overlay, median_ring[i], median_ring[i+1], blue_color, vibrant=False)

    for side_x in [-SIDE_DIST, SIDE_DIST]:
        side_pts = np.array([[side_x, cos(a) * SIDE_RADIUS, sin(a) * SIDE_RADIUS] for a in angles])
        for i in range(len(side_pts)-1):
            draw_3d_segment(overlay, side_pts[i], side_pts[i+1], blue_color, vibrant=False)

    out = img.copy()
    indices = mask > 0
    out[indices] = cv2.addWeighted(img[indices], 0.6, overlay[indices], 0.4, 0)

    for i in range(len(brow_ring)-1):
        if abs(brow_ring[i][0]) <= SIDE_DIST:
            draw_3d_segment(out, brow_ring[i], brow_ring[i+1], blue_color, vibrant=True)
        draw_3d_segment(out, median_ring[i], median_ring[i+1], blue_color, vibrant=True)

    for side_x in [-SIDE_DIST, SIDE_DIST]:
        side_pts = np.array([[side_x, cos(a) * SIDE_RADIUS, sin(a) * SIDE_RADIUS] for a in angles])
        for i in range(len(side_pts)-1):
            draw_3d_segment(out, side_pts[i], side_pts[i+1], blue_color, vibrant=True)
        
        cross = [(np.array([side_x, -SIDE_RADIUS, 0]), np.array([side_x, SIDE_RADIUS, 0])),
                 (np.array([side_x, 0, -SIDE_RADIUS]), np.array([side_x, 0, SIDE_RADIUS]))]
        for p1, p2 in cross:
            draw_3d_segment(out, p1, p2, blue_color, vibrant=True)

    def get_2d(p3d):
        rp = R @ p3d
        return (int(rp[0]*scale + tx), int(rp[1]*scale + ty)), rp[2]

    h_l_2d, z_l = get_2d(hinge_l_3d)
    c_l_2d, _ = get_2d(corner_l_3d)
    if jaw_pts_l and z_l > -0.3:
        full_jaw_l = [h_l_2d, c_l_2d] + jaw_pts_l
        cv2.polylines(out, [np.array(full_jaw_l, np.int32)], False, yellow_color, thickness, cv2.LINE_AA)

    h_r_2d, z_r = get_2d(hinge_r_3d)
    c_r_2d, _ = get_2d(corner_r_3d)
    if jaw_pts_r and z_r > -0.3:
        full_jaw_r = [h_r_2d, c_r_2d] + jaw_pts_r
        cv2.polylines(out, [np.array(full_jaw_r, np.int32)], False, yellow_color, thickness, cv2.LINE_AA)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, blue_color, thickness)

    p_eye_center = R @ np.array([0, 0, 1.0])
    pt_eye_2d = (int(p_eye_center[0]*scale + tx), int(p_eye_center[1]*scale + ty))
    if chin_2d: cv2.line(out, pt_eye_2d, chin_2d, green_color, thickness)
        
    def draw_rotated_tick(center_2d, length_px, color):
        if center_2d is None: return
        half_l = (length_px / scale) / 2
        p1_local, p2_local = np.array([-half_l, 0, 0]), np.array([half_l, 0, 0])
        r1, r2 = R @ p1_local, R @ p2_local
        pt1 = (int(center_2d[0] + r1[0] * scale), int(center_2d[1] + r2[1] * scale))
        pt2 = (int(center_2d[0] + r2[0] * scale), int(center_2d[1] + r2[1] * scale))
        cv2.line(out, pt1, pt2, color, 5) # Thickness 5 as per App.py

    draw_rotated_tick(pt_eye_2d, 25, red_color)
    if nose_2d: draw_rotated_tick(nose_2d, 30, red_color)
    if chin_2d: draw_rotated_tick(chin_2d, 30, red_color)

    return out

# Capture loop
cap = cv2.VideoCapture(0)
smooth_angles = np.zeros(6)
smooth_center = np.zeros(2)

while cap.isOpened():
    success, image = cap.read()
    if not success: break
    image = cv2.flip(image, 1)
    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    results = detector.detect_for_video(mp_image, int(time.time() * 1000))

    if results.face_landmarks:
        for face_landmarks in results.face_landmarks:
            features = normalize_landmarks(face_landmarks, w, h)
            prediction = model(features[np.newaxis], training=False).numpy()[0]
            
            # Smoothing (Specific to Live Mode)
            smooth_angles = (alpha_angles * prediction) + ((1 - alpha_angles) * smooth_angles)
            p, y, r = np.arctan2(smooth_angles[0], smooth_angles[1]), np.arctan2(smooth_angles[2], smooth_angles[3]), np.arctan2(smooth_angles[4], smooth_angles[5])
            
            # Dynamic scale logic from App.py
            eye_dist = np.linalg.norm(np.array([face_landmarks[33].x*w, face_landmarks[33].y*h]) - np.array([face_landmarks[263].x*w, face_landmarks[263].y*h]))
            dynamic_scale = eye_dist * 0.92 * (1.0 / max(cos(y), 0.75))
            
            # Rotation matrix for positioning
            R_full = (np.array([[cos(r), -sin(r), 0], [sin(r), cos(r), 0], [0, 0, 1]]) @ 
                      np.array([[cos(-y), 0, sin(-y)], [0, 1, 0], [-sin(-y), 0, cos(-y)]]) @ 
                      np.array([[1, 0, 0], [0, cos(p), -sin(p)], [0, sin(p), cos(p)]]))
            
            fwd = R_full @ np.array([0, 0, 1.0])
            up  = R_full @ np.array([0, -1.0, 0]) 
            
            # Logic from App.py: Bridge 168 + Fwd Backwards + Up adjustment
            bridge_x, bridge_y = face_landmarks[168].x * w, face_landmarks[168].y * h
            tx_r = bridge_x - (fwd[0] * dynamic_scale) + (up[0] * dynamic_scale * 0.1)
            ty_r = bridge_y - (fwd[1] * dynamic_scale) + (up[1] * dynamic_scale * 0.1)

            smooth_center = (alpha_center * np.array([tx_r, ty_r])) + ((1 - alpha_center) * smooth_center) if np.any(smooth_center) else np.array([tx_r, ty_r])
            
            n_pos = (int(face_landmarks[1].x * w), int(face_landmarks[1].y * h))
            c_pos = (int(face_landmarks[152].x * w), int(face_landmarks[152].y * h))
            j_pts_l = [(int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)) for i in JAW_L_IDX]
            j_pts_r = [(int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)) for i in JAW_R_IDX]

            image = draw_loomis_overlay(image, loomis_vertices, loomis_faces, p, -y, r, 
                                        smooth_center[0], smooth_center[1], scale=dynamic_scale,
                                        nose_2d=n_pos, chin_2d=c_pos, jaw_pts_l=j_pts_l, jaw_pts_r=j_pts_r)

    cv2.imshow('Loomis Head Tracking', image)
    if cv2.waitKey(5) & 0xFF == 27: break

detector.close()
cap.release()
cv2.destroyAllWindows()