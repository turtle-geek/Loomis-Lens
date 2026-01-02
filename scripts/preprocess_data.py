import os
import sys
import glob
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Ensure utils.py is accessible
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(backend_dir)
from utils import normalize_landmarks, get_euler_angles

model_path = os.path.join(backend_dir, 'face_landmarker.task')
data_dir = os.path.join(backend_dir, 'data')

def process_dataset(detector, folder_name, x_file, y_file):
    dataset_path = os.path.join(data_dir, folder_name)
    x_path = os.path.join(data_dir, x_file)
    y_path = os.path.join(data_dir, y_file)
    
    X_data, y_data = [], []
    
    # RESUME LOGIC: Check for existing data
    if os.path.exists(x_path) and os.path.exists(y_path):
        X_data = list(np.load(x_path))
        y_data = list(np.load(y_path))
        print(f"Resuming {folder_name}. Loaded {len(X_data)} existing samples.")

    mat_files = glob.glob(os.path.join(dataset_path, "**/*.mat"), recursive=True)
    start_idx = len(X_data)
    print(f"Processing {folder_name}: {len(mat_files)} total files.")

    try:
        for i in range(start_idx, len(mat_files)):
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
                # Normalization now uses the refined Mean Center + Span logic
                clean_landmarks = normalize_landmarks(detection_result.face_landmarks[0], w, h)
                angles = get_euler_angles(mat_path)
                X_data.append(clean_landmarks)
                y_data.append(angles)
            
            # Periodic Save (Safety Checkpoint every 1000 images)
            if i > 0 and i % 1000 == 0:
                print(f"Progress: {i}/{len(mat_files)} - Auto-saving checkpoint...")
                np.save(x_path, np.array(X_data))
                np.save(y_path, np.array(y_data))

    except KeyboardInterrupt:
        print("\nPaused by user. Saving current progress...")
        np.save(x_path, np.array(X_data))
        np.save(y_path, np.array(y_data))
        sys.exit(0)

    # Final save for the dataset
    X_final, y_final = np.array(X_data), np.array(y_data)
    np.save(x_path, X_final)
    np.save(y_path, y_final)
    print(f"Finished processing {folder_name}. Saved {len(X_final)} samples.")
    return X_final, y_final

if __name__ == "__main__":
    # --- INTERACTIVE SAFETY PROMPT ---
    existing_files = ["X_train_final.npy", "y_train_final.npy", "X_test_final.npy", "y_test_final.npy"]
    found_any = any(os.path.exists(os.path.join(data_dir, f)) for f in existing_files)

    if found_any:
        print("\n" + "!"*40)
        print("EXISTING DATA DETECTED.")
        print("Since you are focusing on Z-axis accuracy, you MUST delete old data.")
        print("!"*40)
        choice = input("Delete existing data and start fresh? (y/n): ").lower()
        
        if choice == 'y':
            for f in existing_files:
                p = os.path.join(data_dir, f)
                if os.path.exists(p): 
                    os.remove(p)
                    print(f"Deleted: {f}")
        else:
            print("Proceeding with Resume Mode (Warning: Normalization might be inconsistent).")

    # Initialize MediaPipe Task
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    # Process 300W-LP (Train) and AFLW2000 (Test)
    process_dataset(detector, '300W_LP', "X_train_final.npy", "y_train_final.npy")
    process_dataset(detector, 'AFLW2000', "X_test_final.npy", "y_test_final.npy")
    print("\n--- Success! Preprocessing Complete ---")