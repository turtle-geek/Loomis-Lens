import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# --- CUSTOM F1 CALLBACK ---
# This calculates your reliability metric after every epoch so we can graph it
class F1MetricCallback(callbacks.Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data
        self.f1_history = []

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.validation_data
        predictions = self.model.predict(X_val, verbose=0)
        error_deg = np.abs(predictions - y_val) * (180.0 / np.pi)
        # Binary target: 1 if all 3 angles are within 5 degrees
        correct_mask = np.all(error_deg < 5.0, axis=1)
        y_true_binary = np.ones(len(y_val))
        y_pred_binary = correct_mask.astype(int)
        
        current_f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        self.f1_history.append(current_f1)
        # This adds 'val_f1' to the console output during training
        print(f" - val_f1: {current_f1:.4f}")

# Setup directory paths
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(backend_dir, 'data')

# 1. LOAD DATASETS
X_train = np.load(os.path.join(data_dir, "X_train_final.npy"))
y_train = np.load(os.path.join(data_dir, "y_train_final.npy"))
X_test = np.load(os.path.join(data_dir, "X_test_final.npy"))
y_test = np.load(os.path.join(data_dir, "y_test_final.npy"))

input_dim = X_train.shape[1]
print(f"Detected Input Dimension: {input_dim}")

# 2. SAMPLE WEIGHTING
def get_sample_weights(labels):
    pitch_intensity = np.abs(labels[:, 0])
    yaw_intensity = np.abs(labels[:, 1])
    weights = 1.0 + (4.0 * pitch_intensity) + (2.0 * yaw_intensity)
    return weights

train_weights = get_sample_weights(y_train)

# 3. MODEL ARCHITECTURE
def residual_block(x, units, dropout_rate):
    shortcut = x
    x = layers.Dense(units, kernel_initializer='lecun_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(units, kernel_initializer='lecun_normal')(x)
    x = layers.BatchNormalization()(x)
    if shortcut.shape[-1] != units:
        shortcut = layers.Dense(units)(shortcut)
    x = layers.Add()([x, shortcut])
    x = layers.Activation('swish')(x)
    return x

def build_model(input_shape):
    inputs = layers.Input(shape=(input_shape,))

    x = layers.GaussianNoise(0.01)(inputs)

    x = layers.Dense(256, kernel_initializer='lecun_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)

    x = residual_block(x, 512, 0.5) # Stronger capacity
    x = residual_block(x, 1024, 0.3)  # Transition
    x = residual_block(x, 1024, 0.4) # Lower dropout (fine tuning)

    x = layers.Dense(256, activation='swish')(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(3, kernel_initializer='zeros')(x)
    return models.Model(inputs, outputs)

model = build_model(input_dim)

# 4. TRAINING SETUP
optimizer = tf.keras.optimizers.AdamW(learning_rate=0.0007, weight_decay=0.01)
model.compile(
    optimizer=optimizer, 
    loss=tf.keras.losses.Huber(delta=1.0), 
    metrics=['mae']
)

early_stop = callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=45, 
    restore_best_weights=True, 
    verbose=1
)

lr_reducer = callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=12, 
    min_lr=1e-7, 
    verbose=1
)

f1_call = F1MetricCallback(validation_data=(X_test, y_test))

# 5. EXECUTE TRAINING
history = model.fit(
    X_train, y_train,
    sample_weight=train_weights,
    epochs=150,
    batch_size=128,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, lr_reducer, f1_call]
)

model.save(os.path.join(backend_dir, "head_pose_model.h5"))

# 6. EVALUATION & GRAPHS
predictions = model.predict(X_test)
error_deg = np.abs(predictions - y_test) * (180.0 / np.pi)
accuracy_5deg = np.mean(np.all(error_deg < 5.0, axis=1)) * 100

print(f"\n--- Final Results ---")
print(f"Accuracy (Within 5.0°): {accuracy_5deg:.2f}%")
print(f"Final Binned F1 Score: {f1_call.f1_history[-1]:.4f}")
print(f"Pitch MAE: {np.mean(error_deg[:, 0]):.2f}°")
print(f"Yaw MAE:   {np.mean(error_deg[:, 1]):.2f}°")

# Create the two graphs
plt.figure(figsize=(14, 6))

# Graph 1: Loss

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss', lw=2)
plt.plot(history.history['val_loss'], label='Val Loss', lw=2)
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Graph 2: F1 Score
plt.subplot(1, 2, 2)
plt.plot(f1_call.f1_history, label='Val F1 (Within 5°)', color='green', lw=2)
plt.title('Binned F1 Score (Reliability)')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.ylim(0, 1)
plt.legend()

plt.tight_layout()
plt.show()