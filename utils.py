import os
import sys
import numpy as np
import scipy.io as sio

def normalize_landmarks(face_landmarks, img_w, img_h):
    """
    Standardizes face landmarks to be position and scale invariant.
    Uses Mean Center for origin and 3D Diagonal Span for scaling to 
    maximize Z-axis (depth) accuracy.
    """
    # Mapping indices from MediaPipe (468) to the extended 77-point set
    REDUCED_77 = [
        162, 234, 93, 58, 172, 136, 150, 149, 152, 378, 379, 365, 397, 288, 323, 454, 389, # Jawline
        71, 63, 105, 66, 107, 336, 296, 334, 293, 301, # Eyebrows
        168, 6, 197, 195, 5, 4, 1, 19, 94, 2, # Nose
        33, 160, 158, 133, 153, 144, # Left Eye
        362, 385, 387, 263, 373, 380, # Right Eye
        61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181, # Outer Lips
        78, 191, 80, 13, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95 # Inner Lips
    ]
    
    # Pixel-space conversion: lm.z is scaled by img_w to keep depth units consistent with X/Y
    full_coords = np.array([[lm.x * img_w, lm.y * img_h, lm.z * img_w] for lm in face_landmarks])
    coords = full_coords[REDUCED_77]

    # STABILITY FIX: Use the 'Geometric Mean' of all 77 points as the origin (0,0,0)
    # This creates a more consistent pivot point for pitch, yaw, and roll.
    center = np.mean(coords, axis=0)
    coords = coords - center

    # SCALE FIX: Use the 3D Diagonal Span (Euclidean distance between max/min bounds)
    # This is much more accurate than width alone when the head tilts up or down.
    dist_span = np.linalg.norm(np.max(coords, axis=0) - np.min(coords, axis=0))

    if dist_span > 0:
        coords = coords / dist_span

    return coords.flatten()

def get_euler_angles(mat_path):
    mat_contents = sio.loadmat(mat_path)
    pose_para = mat_contents['Pose_Para'][0]
    return np.array([pose_para[0], pose_para[1], pose_para[2]]) # pitch, yaw, roll

# Logic to find the backend directory
script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.normpath(os.path.join(script_dir, '..'))
data_dir = os.path.join(backend_dir, 'data')