import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Initialize the AI
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')

# Configure the AI options
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1)

# Create the detector once before the loop
detector = vision.FaceLandmarker.create_from_options(options)

# Set the VideoCapture camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # if image is read unsuccessfully success is set to false
    success, image = cap.read()
    
    if not success:
        break

    # Convert images to RGB format
    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to mediapipe image object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    #Run the detector
    detection_result = detector.detect(mp_image)

    # Draws the results
    # Loop through detected faces.
    if detection_result.face_landmarks:
        for face_landmarks in detection_result.face_landmarks:
            for landmark in face_landmarks:
                # Convert normalized coordinates (0.0 - 1.0) to pixel coordinates
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                
                # Draw a small circle for each point
                cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

    # Displays the captured image output
    cv2.imshow("My video capture", cv2.flip(image, 1))

    # if q is held for 100 miliseconds break
    # & 0xFF ignores everything except for the last 8 bits
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()