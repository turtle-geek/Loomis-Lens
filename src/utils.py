import os
import sys
import numpy as np
import scipy.io as sio

def normalize_landmarks(face_landmarks, img_w, img_h):
    """
    Standardizes face landmarks using a Rigid Anchor (Nose Bridge) and 
    Interocular Scaling. Highly robust for extreme angles.
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
    
    # Pixel-space conversion: lm.z is scaled by img_w to maintain 3D volume consistency
    full_coords = np.array([[lm.x * img_w, lm.y * img_h, lm.z * img_w] for lm in face_landmarks])
    coords = full_coords[REDUCED_77]

    # STABILITY FIX: Use the Nose Bridge (Index 28 in REDUCED_77 / MP 168) as the origin.
    origin_point = coords[28] 
    coords = coords - origin_point

    # SCALE FIX: Interocular Distance (Distance between eye centers)
    # Eye centers calculated from specific indices in the REDUCED_77 set
    left_eye_center = np.mean(coords[37:43], axis=0)
    right_eye_center = np.mean(coords[43:49], axis=0)
    eye_dist = np.linalg.norm(left_eye_center - right_eye_center)

    if eye_dist > 0:
        coords = coords / eye_dist

    return coords.flatten()

def get_euler_angles(mat_path):
    mat_contents = sio.loadmat(mat_path)
    pose_para = mat_contents['Pose_Para'][0]
    return np.array([pose_para[0], pose_para[1], pose_para[2]]) # pitch, yaw, roll

# Path Logic
script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.normpath(os.path.join(script_dir, '..'))
data_dir = os.path.join(backend_dir, 'data')