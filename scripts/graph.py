import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Paths
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(backend_dir, 'data')

# 1. Load the original raw data
X_train = np.load(os.path.join(data_dir, "X_train_final.npy"))
X_test = np.load(os.path.join(data_dir, "X_test_final.npy"))
y_test = np.load(os.path.join(data_dir, "y_test_final.npy"))

# 2. Recreate the EXACT scaling from the training day
X_mean = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0) + 1e-7
X_test_scaled = (X_test - X_mean) / X_std

# 3. Load the old model
model = tf.keras.models.load_model(os.path.join(backend_dir, "head_pose_model.h5"), compile=False)

# 4. Predict
predictions = model.predict(X_test_scaled)

# 5. The "Old Way" Calculation (Matching your 53% run)
# Your old script compared raw values to 5.0 directly
error_raw = np.abs(predictions - y_test)
acc_5_old_logic = np.mean(np.all(error_raw < 5.0, axis=1)) * 100

# The "Real Degree" Calculation (The honest truth)
# Assuming your y_test is in Radians
error_deg = error_raw * (180.0 / np.pi)
acc_5_real = np.mean(np.all(error_deg < 5.0, axis=1)) * 100

print(f"\n--- Recovery Results ---")
print(f"Accuracy (Using Old Logic): {acc_5_old_logic:.2f}%")
print(f"Accuracy (REAL Degrees): {acc_5_real:.2f}%")
print(f"Real Pitch MAE: {np.mean(error_deg[:,0]):.2f}°")

# 6. GRAPHS
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_test[:, 0], predictions[:, 0], alpha=0.1, color='red')
plt.title("Pitch Correlation (Raw Values)")
plt.subplot(1, 2, 2)
plt.hist(error_deg.flatten(), bins=50, color='blue', alpha=0.7)
plt.axvline(5.0, color='red', linestyle='--', label='5° Mark')
plt.title("Error Distribution in Real Degrees")
plt.legend()
plt.show()