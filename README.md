# Loomis Lens
An interactive tool that uses real-time facial landmark recognition and head pose estimation to project 3D anatomical overlays. It applies the Loomis Method to live webcam feeds or images to help artists visualize head planes and perspective.

## Table of Contents
- [How it Works](#how-it-works)
- [Model Performance](#model-performance)
- [Dataset and Preprocessing](#dataset-and-preprocessing)
- [Model Architecture](#model-architecture)
- [Training Details](#training-details)
- [3D Asset Generation](#3d-asset-generation)
- [Deployment](#deployment)

## How it Works
The application functions as a digital assistant for anatomical study through a three-stage pipeline:
1. **Extraction:** Extracts 468 3D facial landmarks from a camera feed or uploaded image using MediaPipe.
2. **Estimation:** A deep neural network predicts rotational vectors (Sine and Cosine values for Yaw, Pitch, and Roll) to ensure mathematical continuity.
3. **Projection:** A custom-modeled Loomis mesh is projected onto the user's face using a pinhole camera model and perspective mapping.



## Model Performance
The model was evaluated on the **AFLW2000-3D** benchmark set and significantly outperforms general-purpose landmark estimators:

| Metric | Performance | Context |
| :--- | :--- | :--- |
| **Mean Absolute Error (MAE)** | **4.04°** | Outperforms standard landmark-based estimators. |
| **Robust Accuracy (<10°)** | **93.36%** | Ensures a stable, "jitter-free" overlay for drawing. |
| **Strict Accuracy (<5°)** | **58.01%** | Demonstrates high-precision tracking for detailed anatomical study. |

## Dataset and Preprocessing
* **Source Data**: Trained on the **300W-LP** and **AFLW-3D** datasets.
* **Augmentation**: Implements real-time mirror augmentation, inverting Yaw and Roll angles to improve model generalization.
* **Normalization**: Landmarks are centered at the nose tip and scaled by interocular distance to achieve scale invariance.

## Model Architecture
The final model is a custom Deep Neural Network implemented in TensorFlow with a focus on geometric precision:
* **Input Processing**: Uses **Gaussian Noise** (0.005) to improve robustness against shaky webcam feeds.
* **Core Blocks**: Features three **Residual Blocks** that utilize skip connections to maintain gradient stability during deep training.
* **Attention Mechanism**: Integrated **Squeeze-and-Excitation (SE) blocks** allow the model to dynamically re-weight facial landmark features.
* **Activation**: Uses the **Swish** activation function for smoother mathematical curves compared to standard ReLU.

## Training Details
This project implements a robust machine learning pipeline designed for high-accuracy regression:
* **Target Representation**: The model predicts **Sine and Cosine** values for each axis to avoid the "Gimbal Lock" problem and ensure mathematical continuity.
* **Loss Function**: A hybrid **Huber Loss** ($\delta=2.0$) combined with a **Margin Penalty** ($\text{threshold}=5.0^\circ$) to penalize outliers while remaining robust to noise.
* **Sample Weighting**: Uses a custom weighting scheme to prioritize accurate **Pitch** and **Yaw** estimation, which are critical for anatomical alignment.
* **Optimizer**: **AdamW** (Learning Rate: $3 \times 10^{-4}$, Weight Decay: 0.04) for improved weight regularization.
* **Training Strategy**: Utilizes **Learning Rate Reduction on Plateau** and **Early Stopping** (patience: 35) to find the global minimum without overfitting.



## 3D Asset Generation
The Loomis head overlay was custom-modeled in Blender to ensure strict anatomical accuracy:
* **Blender Workflow**: A 3D mesh was generated following the proportions of the Loomis Method.
* **Rendering**: Optimized as a low-poly `.obj` file.
* **Stability**: The frontend uses **Render Damping** (0.9) to smooth the 3D guide's movement during real-time tracking.



## Deployment
* **Frontend**: Hosted on **Vercel** at [loomis-lens.vercel.app](https://loomis-lens.vercel.app).
* **Backend**: Python API containerized via **Docker** and deployed on **AWS Lambda/ECR** for scalable, serverless inference.
