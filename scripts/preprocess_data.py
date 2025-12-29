import scipy.io as sio
import numpy as np
import os
import glob

def get_euler_angles(mat_path):
    # Load the .mat file
    mat_contents = sio.loadmat(mat_path)
    
    # Pose_Para contains [pitch, yaw, roll, tdx, tdy, tdz, scale]
    # We only want the first 3 (Pitch, Yaw, Roll)
    pose_para = mat_contents['Pose_Para'][0]
    pitch = pose_para[0]
    yaw = pose_para[1]
    roll = pose_para[2]
    
    return np.array([pitch, yaw, roll])

if __name__ == "__main__":
    dataset_path = "../data/300W_LP"
    mat_files = glob.glob(os.path.join(dataset_path, "**/*.mat"), recursive = True)

    for i in range(min(10, len(mat_files))):
        file_path = mat_files[i]
        try:
            angles = get_euler_angles(file_path)
            file_name = os.path.basename(file_path)
            print(f"{i+1}. {file_name} -> Pitch: {angles[0]:.4f}, Yaw: {angles[1]:.4f}, Roll: {angles[2]:.4f}")
        except Exception as e:
                print(f"Could not read {file_path}: {e}")