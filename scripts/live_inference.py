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

# --- PATH SETUP ---
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(backend_dir)
from utils import normalize_landmarks

# --- MODEL INITIALIZATION ---
model = tf.keras.models.load_model(
    os.path.join(backend_dir, "head_pose_model_2.h5"),
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

# --- MESH LOADING ---
def load_loomis_mesh_with_faces(file_path):
    vertices, faces = [], []
    if not os.path.exists(file_path):
        return np.array([]), []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '): vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith('f '): faces.append([int(i.split('/')[0]) - 1 for i in line.split()[1:]])
    return np.array(vertices), faces

mesh_path = os.path.join(backend_dir, "assets", "loomis_base.obj")
loomis_vertices, loomis_faces = load_loomis_mesh_with_faces(mesh_path)

# --- RENDERING FUNCTIONS ---
def draw_brow_axes(img, R, origin_2d, size=50):
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

    temp_img = img.copy()

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
            overlay_face = temp_img.copy()
            cv2.fillPoly(overlay_face, [pts], (240, 240, 240)) 
            cv2.polylines(overlay_face, [pts], True, (180, 180, 180), 1)
            cv2.addWeighted(overlay_face, 0.4, temp_img, 0.6, 0, temp_img)

    num_segments = 128
    angles = np.linspace(0, 2 * np.pi, num_segments)
    brow_pts = np.array([[cos(a), 0, sin(a)] for a in angles])
    median_pts = np.array([[0, cos(a), sin(a)] for a in angles])
    side_l_pts = np.array([[-SIDE_DIST, cos(a) * SIDE_RADIUS, sin(a) * SIDE_RADIUS] for a in angles])
    side_r_pts = np.array([[SIDE_DIST, cos(a) * SIDE_RADIUS, sin(a) * SIDE_RADIUS] for a in angles])

    def draw_line_3d(line_pts, color, is_ring=True):
        for i in range(len(line_pts) - 1):
            p1_orig, p2_orig = line_pts[i], line_pts[i+1]
            if is_ring:
                if abs(p1_orig[0]) > SIDE_DIST or abs(p2_orig[0]) > SIDE_DIST: continue
                if abs(p1_orig[1]) > SIDE_RADIUS or abs(p2_orig[1]) > SIDE_RADIUS: continue

            p1, p2 = R @ p1_orig, R @ p2_orig
            if p1[2] > -0.1:
                cv2.line(temp_img, (int(p1[0]*scale+tx), int(p1[1]*scale+ty)), 
                         (int(p2[0]*scale+tx), int(p2[1]*scale+ty)), color, 2)

    draw_line_3d(brow_pts, (255, 255, 0))   
    draw_line_3d(median_pts, (255, 0, 255)) 
    draw_line_3d(side_l_pts, (0, 255, 255), False) 
    draw_line_3d(side_r_pts, (0, 255, 255), False) 

    if chin_target_2d is not None and brow_target_2d is not None:
        cv2.line(temp_img, brow_target_2d, chin_target_2d, (255, 255, 255), 2)

    cv2.circle(temp_img, (int(tx), int(ty)), 5, (0, 255, 0), -1)
    return temp_img

# --- CAPTURE LOOP ---
cap = cv2.VideoCapture(0)
smooth_angles, alpha = np.zeros(3), 0.6 

while cap.isOpened():
    success, image = cap.read()
    if not success: break
    image = cv2.flip(image, 1); h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    results = detector.detect_for_video(mp_image, int(time.time() * 1000))

    if results.face_landmarks:
        overlay_layer = image.copy()
        for face_landmarks in results.face_landmarks:
            # 1. Feature Extraction (No longer slicing; supports full 231 features)
            features = normalize_landmarks(face_landmarks, w, h)

            # 2. Prediction & Smoothing
            prediction = model(features[np.newaxis], training=False).numpy()[0]
            smooth_angles = (alpha * prediction) + ((1 - alpha) * smooth_angles)
            pitch, yaw, roll = smooth_angles
            current_yaw = -yaw 

            # 3. Anchoring and Scale Logic (Remains identical to head-base logic)
            brow_target = face_landmarks[168]   
            hairline = face_landmarks[10]     
            nose_base = face_landmarks[2]     
            chin_landmark = face_landmarks[152]

            bx, by = brow_target.x * w, brow_target.y * h

            dist_to_top = np.linalg.norm(np.array([bx, by]) - np.array([hairline.x*w, hairline.y*h]))
            dist_to_bottom = np.linalg.norm(np.array([bx, by]) - np.array([nose_base.x*w, nose_base.y*h]))
            
            total_vertical_span = dist_to_top + dist_to_bottom
            dynamic_scale = total_vertical_span / 1.25

            # 4. Rotation Matrix for Anchor
            R_x = np.array([[1, 0, 0], [0, cos(pitch), -sin(pitch)], [0, sin(pitch), cos(pitch)]])
            R_y = np.array([[cos(current_yaw), 0, sin(current_yaw)], [0, 1, 0], [-sin(current_yaw), 0, cos(current_yaw)]])
            R_z = np.array([[cos(roll), -sin(roll), 0], [sin(roll), cos(roll), 0], [0, 0, 1]])
            R_full = R_z @ R_y @ R_x

            surface_vector = R_full @ np.array([0, 0, 1.0])
            down_vector = R_full @ np.array([0, 1.0, 0])
            
            tx = bx - (surface_vector[0] * dynamic_scale * 0.95) - (down_vector[0] * dynamic_scale * 0.05)
            ty = by - (surface_vector[1] * dynamic_scale * 0.95) - (down_vector[1] * dynamic_scale * 0.05)

            brow_2d = (int(bx), int(by))
            chin_2d = (int(chin_landmark.x * w), int(chin_landmark.y * h))

            # 5. RENDER
            overlay_layer = draw_solid_loomis_overlay(overlay_layer, loomis_vertices, loomis_faces, 
                                               pitch, current_yaw, roll, 
                                               tx, ty, scale=dynamic_scale,
                                               chin_target_2d=chin_2d,
                                               brow_target_2d=brow_2d)

            draw_brow_axes(overlay_layer, R_full, brow_2d, size=dynamic_scale * 0.5)

        image = cv2.addWeighted(overlay_layer, 0.6, image, 0.4, 0)
    
    cv2.imshow('Loomis Lens - 77 Landmarks', image)
    if cv2.waitKey(5) & 0xFF == 27: break

detector.close()
cap.release()
cv2.destroyAllWindows()