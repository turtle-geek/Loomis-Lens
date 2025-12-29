import numpy as np

def normalize_landmarks(face_landmarks):
    """
    Standardizes face landmarks to be position and scale invariant

    This function shifts the face so the nose is at (0,0,0) and resizes
    the landmarks so the face width is 1.0 unit

    Args:
        face_landmarks: A list of landmarks from MediaPipe FaceMesh.

    Returns:
        np.array: A flattened 1D array of 1404 normalized coordinates 
    """
    # Convert MediaPipe landmarks to a NumPy array (x, y, z)
    coords_list = []
    for lm in face_landmarks:
        point = [lm.x, lm.y, lm.z]
        coords_list.append(point)
    
    coords = np.array(coords_list)

    # Use nose tip as the origin
    nose_tip = coords[1] 
    coords = coords - nose_tip

    # Find distance between left and right side of face
    left_side = coords[127]
    right_side = coords[356]
    face_width = np.linalg.norm(left_side - right_side)

    # Divide all coordinates by width to scale the face to 1.0 unit
    if face_width > 0:
        coords = coords / face_width

    flattended = coords.flatten()

    # Restrict to first 1404 values
    return flattended[:1404]