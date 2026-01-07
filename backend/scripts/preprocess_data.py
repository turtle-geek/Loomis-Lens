import os
import sys
import glob
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Get the root directory
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

# Import normalization logic from the src package
from src.utils import normalize_landmarks, get_euler_angles

# Update paths to point to the new folder organization
model_path = os.path.join(backend_dir, 'models', 'face_landmarker.task')
data_dir = os.path.join(backend_dir, 'data')

def process_dataset(detector, folder_name, x_file, y_file):
    dataset_path = os.path.join(data_dir, folder_name)
    x_path = os.path.join(data_dir, x_file)
    y_path = os.path.join(data_dir, y_file)
    
    X_data, y_data = [], []
    
    if os.path.exists(x_path) and os.path.exists(y_path):
        X_data = list(np.load(x_path))
        y_data = list(np.load(y_path))
        print(f"Resuming {folder_name}. Loaded {len(X_data)} samples.")

    # Search for mat files in the specific dataset subfolder
    mat_files = glob.glob(os.path.join(dataset_path, "**/*.mat"), recursive=True)
    processed_files_count = len(X_data) // 2 
    print(f"Processing {folder_name}: {len(mat_files)} total files.")

    try:
        for i in range(processed_files_count, len(mat_files)):
            mat_path = mat_files[i]
            img_path = mat_path.replace('.mat', '.jpg')
            if not os.path.exists(img_path): continue

            image = cv2.imread(img_path)
            if image is None: continue
            h, w, _ = image.shape
            
            rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = detector.detect(mp_image)

            if detection_result.face_landmarks:
                # Original data
                clean_landmarks = normalize_landmarks(detection_result.face_landmarks[0], w, h)
                angles = get_euler_angles(mat_path)
                X_data.append(clean_landmarks)
                y_data.append(angles)

                # Mirror augmentation
                mirrored_lms = clean_landmarks.reshape(-1, 3).copy()
                mirrored_lms[:, 0] *= -1 # Flip x axis
                
                # Invert yaw and roll for mirrored image
                mirrored_angles = np.array([angles[0], -angles[1], -angles[2]])
                
                X_data.append(mirrored_lms.flatten())
                y_data.append(mirrored_angles)
            
            if i > 0 and i % 500 == 0:
                print(f"Progress: {i}/{len(mat_files)} (Samples: {len(X_data)}) - Saving...")
                np.save(x_path, np.array(X_data))
                np.save(y_path, np.array(y_data))

    except KeyboardInterrupt:
        print("\nProcess interrupted. Saving...")
        np.save(x_path, np.array(X_data))
        np.save(y_path, np.array(y_data))
        sys.exit(0)

    np.save(x_path, np.array(X_data))
    np.save(y_path, np.array(y_data))
    return np.array(X_data), np.array(y_data)

if __name__ == "__main__":
    # Ensure data directory exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    existing_files = ["X_train_final.npy", "y_train_final.npy", "X_test_final.npy", "y_test_final.npy"]
    if any(os.path.exists(os.path.join(data_dir, f)) for f in existing_files):
        print("\n" + "!"*50)
        print("New normalization detected")
        print("Delete old npy files for best results")
        print("!"*50)
        if input("Delete and start fresh? (y/n): ").lower() == 'y':
            for f in existing_files:
                p = os.path.join(data_dir, f)
                if os.path.exists(p): os.remove(p)

    # Initialize mediapipe from the models folder
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options, 
        num_faces=1, 
        min_face_presence_confidence=0.5
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    # Dataset subfolders should be located in data 300w_lp and data aflw2000
    process_dataset(detector, '300W_LP', "X_train_final.npy", "y_train_final.npy")
    process_dataset(detector, 'AFLW2000', "X_test_final.npy", "y_test_final.npy")
    print("\n--- Preprocessing Complete ---")