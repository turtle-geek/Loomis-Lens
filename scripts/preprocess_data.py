import os
import sys
import glob
import numpy as np
import cv2
import mediapipe as mp
import scipy.io as sio
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(backend_dir)
from utils import normalize_landmarks

model_path = os.path.join(backend_dir, 'face_landmarker.task')
dataset_path = os.path.join(backend_dir, 'data', '300W_LP')

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1)

detector = vision.FaceLandmarker.create_from_options(options)

def get_euler_angles(mat_path):
    mat_contents = sio.loadmat(mat_path)
    
    # Pose_Para contains [pitch, yaw, roll, tdx, tdy, tdz, scale]
    pose_para = mat_contents['Pose_Para'][0]
    pitch = pose_para[0]
    yaw = pose_para[1]
    roll = pose_para[2]
    
    return np.array([pitch, yaw, roll])

def process_dataset(dataset_path, limit=100):
    X_data = []
    y_data = []
    
    mat_files = glob.glob(os.path.join(dataset_path, "**/*.mat"), recursive=True)
    print(f"Found {len(mat_files)} files. Processing first {limit}...")

    for i in range(min(limit, len(mat_files))):
        mat_path = mat_files[i]
        img_path = mat_path.replace('.mat', '.jpg')

        if not os.path.exists(img_path):
            continue

        # Load image and convert for MediaPipe Tasks
        image = cv2.imread(img_path)
        # Convert BGR image to RGB
        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Convert the RGB image to mediapipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Run Detector
        detection_result = detector.detect(mp_image)

        # Only save if a face was actually found
        if detection_result.face_landmarks:
            clean_landmarks = normalize_landmarks(detection_result.face_landmarks[0])
            angles = get_euler_angles(mat_path)
            
            X_data.append(clean_landmarks)
            y_data.append(angles)
            
        if i % 10 == 0:
            print(f"Progress: {i}/{limit}")

    return np.array(X_data), np.array(y_data)

if __name__ == "__main__":
    # Process a small batch to test
    X, y = process_dataset(dataset_path, limit=50)
    
    # Save the processed data
    # These will be created in your backend/data/ folder
    np.save("../data/X_train_small.npy", X)
    np.save("../data/y_train_small.npy", y)
    
    print(f"Finished! Saved {len(X)} samples.")
    print("X shape:", X.shape) # Should be (N, 1404)
    print("y shape:", y.shape) # Should be (N, 3)