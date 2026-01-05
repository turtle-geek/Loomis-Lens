import os
import sys
import numpy as np
import scipy.io as sio

def normalize_landmarks(face_landmarks, img_w, img_h):
    """
    Standardizes face landmarks using a rigid anchor and 
    interocular scaling for extreme angle robustness
    """
    # Reduced set of 77 landmarks for faster processing
    REDUCED_77 = [
        162, 234, 93, 58, 172, 136, 150, 149, 152, 378, 379, 365, 397, 288, 323, 454, 389, # Jawline
        71, 63, 105, 66, 107, 336, 296, 334, 293, 301, # Eyebrows
        168, 6, 197, 195, 5, 4, 1, 19, 94, 2, # Nose
        33, 160, 158, 133, 153, 144, # Left eye
        362, 385, 387, 263, 373, 380, # Right eye
        61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181, # Outer lips
        78, 191, 80, 13, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95 # Inner lips
    ]
    
    # Convert mediapipe coordinates to pixel space
    full_coords = np.array([[lm.x * img_w, lm.y * img_h, lm.z * img_w] for lm in face_landmarks])
    coords = full_coords[REDUCED_77]

    # Set the nose bridge as the center point for rotation
    origin_point = coords[28] 
    coords = coords - origin_point

    # Calculate distance between eyes for scale normalization
    left_eye_center = np.mean(coords[37:43], axis=0)
    right_eye_center = np.mean(coords[43:49], axis=0)
    eye_dist = np.linalg.norm(left_eye_center - right_eye_center)

    # Scale landmarks so models are head size independent
    if eye_dist > 0:
        coords = coords / eye_dist

    return coords.flatten()

def get_euler_angles(mat_path):
    # Extract pitch yaw and roll from the dataset matlab files
    mat_contents = sio.loadmat(mat_path)
    pose_para = mat_contents['Pose_Para'][0]
    return np.array([pose_para[0], pose_para[1], pose_para[2]])

# Directory setup
script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.normpath(os.path.join(script_dir, '..'))
data_dir = os.path.join(backend_dir, 'data')