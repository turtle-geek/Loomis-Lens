import sys
import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
from fastapi import FastAPI, UploadFile, File, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from math import cos, sin, sqrt

os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

# Path setup
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from src.utils import normalize_landmarks

# Configuration
RENDER_DAMPING = 0.9  
JAW_L_IDX = [58, 172, 136, 150, 149, 176, 148, 152]
JAW_R_IDX = [288, 397, 365, 379, 378, 400, 377, 152]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and detector initialization
try:
    model = tf.keras.models.load_model(
        os.path.join(backend_dir, "models", "head_pose_model.h5"), 
        compile=False
    )
    
    model_path = os.path.join(backend_dir, "models", "face_landmarker.task")
    detector = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1
        )
    )
    print("SUCCESS: Models loaded correctly.")

except Exception as e:
    # This prints error in CloudWatch logs
    print(f"CRITICAL ERROR during initialization: {str(e)}")
    raise e

def load_loomis_mesh(file_path):
    vertices, faces = [], []
    if not os.path.exists(file_path):
        return np.array([]), []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '): 
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith('f '): 
                faces.append([int(i.split('/')[0]) - 1 for i in line.split()[1:]])
    return np.array(vertices), faces

mesh_path = os.path.join(backend_dir, "assets", "loomis_base.obj")
loomis_vertices, loomis_faces = load_loomis_mesh(mesh_path)

def get_rotation_matrix(p, y, r):
    R_x = np.array([[1, 0, 0], [0, cos(p), -sin(p)], [0, sin(p), cos(p)]])
    R_y = np.array([[cos(y), 0, sin(y)], [0, 1, 0], [-sin(y), 0, cos(y)]])
    R_z = np.array([[cos(r), -sin(r), 0], [sin(r), cos(r), 0], [0, 0, 1]])
    return R_z @ R_y @ R_x

def draw_loomis_overlay(img, mesh_points, faces, pitch, yaw, roll, tx, ty, scale=100, 
                        nose_2d=None, chin_2d=None, jaw_pts_l=None, jaw_pts_r=None):
    SIDE_DIST = 0.7454 
    SIDE_RADIUS = sqrt(1.0 - SIDE_DIST**2)
    
    # BGRA (Blue, Green, Red, Alpha) to support transparency
    COLORS = {
        'front': (255, 120, 0, 255),    # Solid Orange
        'back': (180, 100, 50, 150),    # Semi-transparent Muted Blue
        'red': (0, 0, 255, 255),        # Solid Red
        'green': (0, 255, 0, 255),      # Solid Green
        'yellow': (0, 255, 255, 255),   # Solid Yellow
        'mesh_fill': (242, 242, 242, 80) # Faint gray fill for the head volume
    }
    
    R = get_rotation_matrix(pitch * RENDER_DAMPING, yaw * RENDER_DAMPING, roll * RENDER_DAMPING)
    
    # Create a 4-channel transparent overlay instead of modifying the original image
    # Shape is (height, width, 4), dtype is uint8. All pixels start at 0 (fully transparent)
    overlay = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    if len(mesh_points) > 0:
        transformed = []
        for p in mesh_points:
            rp = R @ np.array(p)
            transformed.append([int(rp[0] * scale + tx), int(rp[1] * scale + ty), rp[2]])
        transformed = np.array(transformed)

        # Render depth sorted faces for the background mesh volume
        face_depths = sorted([(i, np.mean(transformed[face, 2])) for i, face in enumerate(faces)], key=lambda x: x[1])
        for face_idx, _ in face_depths:
            pts = transformed[faces[face_idx], :2].astype(np.int32)
            cv2.fillPoly(overlay, [pts], COLORS['mesh_fill'])
            cv2.fillPoly(mask, [pts], 255)

    def draw_3d_segment(target, p1, p2, color, vibrant):
        r1, r2 = R @ p1, R @ p2
        is_front = (r1[2] > -0.1 or r2[2] > -0.1)
        if vibrant == is_front:
            pt1 = (int(r1[0] * scale + tx), int(r1[1] * scale + ty))
            pt2 = (int(r2[0] * scale + tx), int(r2[1] * scale + ty))
            # cv2.line in 4-channel targets respects the alpha provided in the color tuple
            cv2.line(target, pt1, pt2, color, 2, cv2.LINE_AA)

    angles = np.linspace(0, 2 * np.pi, 120)
    rings = [
        np.array([[cos(a), 0, sin(a)] for a in angles]), # Brow
        np.array([[0, cos(a), sin(a)] for a in angles])  # Median
    ]
    
    # Render background guides for circles and side crosses
    for ring in rings:
        for i in range(len(ring)-1):
            if ring is rings[1] or abs(ring[i][0]) <= SIDE_DIST:
                draw_3d_segment(overlay, ring[i], ring[i+1], COLORS['back'], False)

    for side_x in [-SIDE_DIST, SIDE_DIST]:
        side_pts = np.array([[side_x, cos(a) * SIDE_RADIUS, sin(a) * SIDE_RADIUS] for a in angles])
        cross = [(np.array([side_x, -SIDE_RADIUS, 0]), np.array([side_x, SIDE_RADIUS, 0])),
                 (np.array([side_x, 0, -SIDE_RADIUS]), np.array([side_x, 0, SIDE_RADIUS]))]
        for i in range(len(side_pts)-1):
            draw_3d_segment(overlay, side_pts[i], side_pts[i+1], COLORS['back'], False)
        for p1, p2 in cross:
            draw_3d_segment(overlay, p1, p2, COLORS['back'], False)

    # Render foreground guides for circles and side crosses directly onto overlay
    for ring in rings:
        for i in range(len(ring)-1):
            if ring is rings[1] or abs(ring[i][0]) <= SIDE_DIST:
                draw_3d_segment(overlay, ring[i], ring[i+1], COLORS['front'], True)

    for side_x in [-SIDE_DIST, SIDE_DIST]:
        side_pts = np.array([[side_x, cos(a) * SIDE_RADIUS, sin(a) * SIDE_RADIUS] for a in angles])
        cross = [(np.array([side_x, -SIDE_RADIUS, 0]), np.array([side_x, SIDE_RADIUS, 0])),
                 (np.array([side_x, 0, -SIDE_RADIUS]), np.array([side_x, 0, SIDE_RADIUS]))]
        for i in range(len(side_pts)-1):
            draw_3d_segment(overlay, side_pts[i], side_pts[i+1], COLORS['front'], True)
        for p1, p2 in cross:
            draw_3d_segment(overlay, p1, p2, COLORS['front'], True)

    def get_2d(p3d):
        rp = R @ p3d
        return (int(rp[0]*scale + tx), int(rp[1]*scale + ty))

    # Render yellow jawline connections
    hinges = [np.array([-SIDE_DIST, SIDE_RADIUS * 0.5, 0.0]), np.array([SIDE_DIST, SIDE_RADIUS * 0.5, 0.0])]
    corners = [np.array([-SIDE_DIST, SIDE_RADIUS * 0.9, 0.1]), np.array([SIDE_DIST, SIDE_RADIUS * 0.9, 0.1])]
    for i, pts in enumerate([jaw_pts_l, jaw_pts_r]):
        if pts:
            full_jaw = [get_2d(hinges[i]), get_2d(corners[i])] + pts
            cv2.polylines(overlay, [np.array(full_jaw, np.int32)], False, COLORS['yellow'], 2, cv2.LINE_AA)

    # Add vibrant outline to the face mesh area on the overlay
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, COLORS['front'], 2, cv2.LINE_AA)

    # Calculate center of the eye line
    pt_eye_2d = get_2d(np.array([0, 0, 1.0]))
    if chin_2d: cv2.line(overlay, pt_eye_2d, chin_2d, COLORS['green'], 2, cv2.LINE_AA)
        
    def draw_tick(center, length_factor, color):
        if center is None: return
        actual_length = scale * length_factor
        half = (actual_length / scale) / 2
        r1, r2 = R @ np.array([-half, 0, 0]), R @ np.array( [half, 0, 0])
        cv2.line(overlay, (int(center[0] + r1[0] * scale), int(center[1] + r1[1] * scale)),
                          (int(center[0] + r2[0] * scale), int(center[1] + r2[1] * scale)), color, 5, cv2.LINE_AA)

    # Drawing red ticks
    tick_size = 0.25 
    draw_tick(pt_eye_2d, tick_size, COLORS['red'])
    for p in [nose_2d, chin_2d]: draw_tick(p, tick_size, COLORS['red'])

    return overlay

@app.post("/generate-overlay")
async def generate_overlay(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None: raise HTTPException(status_code=400, detail="Invalid image file")

    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    results = detector.detect(mp_image)
    if not results.face_landmarks: return {"head_detected": False, "message": "No head detected"}

    face = results.face_landmarks[0]
    features = normalize_landmarks(face, w, h)
    pred = model(features[np.newaxis], training=False).numpy()[0]

    p, y, r = np.arctan2(pred[0], pred[1]), np.arctan2(pred[2], pred[3]), np.arctan2(pred[4], pred[5])
    eye_dist = np.linalg.norm(np.array([face[33].x*w, face[33].y*h]) - np.array([face[263].x*w, face[263].y*h]))
    scale = eye_dist * 0.92 * (1.0 / max(cos(y), 0.75))

    R_full = get_rotation_matrix(p, -y, r)
    fwd, up = R_full @ [0, 0, 1], R_full @ [0, -1, 0]
    tx = (face[168].x * w) - (fwd[0] * scale) + (up[0] * scale * 0.1)
    ty = (face[168].y * h) - (fwd[1] * scale) + (up[1] * scale * 0.1)

    # The function now returns ONLY the overlay lines with transparency
    res = draw_loomis_overlay(image, loomis_vertices, loomis_faces, p, -y, r, tx, ty, scale,
                              (int(face[2].x * w), int(face[2].y * h)), (int(face[152].x * w), int(face[152].y * h)),
                              [(int(face[i].x * w), int(face[i].y * h)) for i in JAW_L_IDX],
                              [(int(face[i].x * w), int(face[i].y * h)) for i in JAW_R_IDX])

    # Encode as PNG (essential for transparency)
    _, encoded = cv2.imencode('.png', res)
    return Response(content=encoded.tobytes(), media_type="image/png")

@app.get("/")
def read_root(): return {"status": "Loomis Lens API Active"}

@app.get("/health")
def health():
    return {"status": "awake"}

handler = Mangum(app)