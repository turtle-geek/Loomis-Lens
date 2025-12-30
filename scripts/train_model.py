import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(backend_dir, 'data')

X = np.load(os.path.join(data_dir, "X_train_5k.npy"))
y = np.load(os.path.join(data_dir, "y_train_5k.npy"))

# Split into training and testing sets (80% training and 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)

model = models.Sequential([
    layers.Input(shape=(1404,)), # 468 landmarks * 3
    layers.Dense(512, activation='relu'), # Learns basic shapes
    layers.Dropout(0.2), # Prevents model from memorizing
    layers.Dense(256, activation='relu'), # Learns 3D spacial relationships
    layers.Dense(128, activation='relu'), # Refines data into rotation signals
    layers.Dense(3) # Output: Pitch, Yaw, Roll
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("Starting training...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
)

model_save_path = os.path.join(backend_dir, "head_pose_model.h5")
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()