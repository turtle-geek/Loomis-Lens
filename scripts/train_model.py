import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# --- CUSTOM METRICS & LOSS ---

def weighted_pose_loss(y_true, y_pred):
    """
    Uses Huber Loss to handle noisy landmarks and weights Pitch 3.0x.
    """
    huber = tf.keras.losses.Huber(delta=0.1) # Tight delta for high precision
    
    # Per-angle loss calculation
    loss_p = huber(y_true[:, 0], y_pred[:, 0]) * 3.0 # Stronger Pitch focus
    loss_y = huber(y_true[:, 1], y_pred[:, 1]) * 1.0
    loss_r = huber(y_true[:, 2], y_pred[:, 2]) * 1.0
    
    return loss_p + loss_y + loss_r

class AccuracyF1Callback(callbacks.Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data
        self.f1_history = []
        self.acc_history = []

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.validation_data
        predictions = self.model.predict(X_val, verbose=0)
        
        # Reliability check in degrees
        error_deg = np.abs(predictions - y_val) * (180.0 / np.pi)
        
        # Accuracy: All 3 angles must be within 5 degrees
        correct_mask = np.all(error_deg < 5.0, axis=1)
        accuracy = np.mean(correct_mask) * 100
        
        # Per-angle accuracy for debugging
        p_acc = np.mean(error_deg[:, 0] < 5.0) * 100
        y_acc = np.mean(error_deg[:, 1] < 5.0) * 100
        
        # F1 Score
        y_true_binary = np.ones(len(y_val))
        y_pred_binary = correct_mask.astype(int)
        current_f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        
        self.f1_history.append(current_f1)
        self.acc_history.append(accuracy)
        
        print(f" - val_acc_5deg: {accuracy:.2f}% | P-Acc: {p_acc:.1f}% | Y-Acc: {y_acc:.1f}% | val_f1: {current_f1:.4f}")

# Setup paths
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(backend_dir, 'data')

# 1. LOAD DATASETS
X_train = np.load(os.path.join(data_dir, "X_train_final.npy"))
y_train = np.load(os.path.join(data_dir, "y_train_final.npy"))
X_test = np.load(os.path.join(data_dir, "X_test_final.npy"))
y_test = np.load(os.path.join(data_dir, "y_test_final.npy"))

input_dim = X_train.shape[1]
print(f"Input Dim: {input_dim} | Training on {len(X_train)} samples")

# 2. SAMPLE WEIGHTING
def get_sample_weights(labels):
    # Focus heavily on non-zero pitch intensities
    pitch_intensity = np.abs(labels[:, 0])
    yaw_intensity = np.abs(labels[:, 1])
    weights = 1.0 + (6.0 * pitch_intensity) + (2.5 * yaw_intensity)
    return weights

train_weights = get_sample_weights(y_train)

# 3. ADVANCED ARCHITECTURE (SE-RESNET STYLE)

def se_block(inputs, ratio=8):
    filters = inputs.shape[-1]
    se = layers.Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(inputs)
    se = layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    return layers.Multiply()([inputs, se])

def residual_block(x, units, dropout_rate):
    shortcut = x
    x = layers.Dense(units, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(units, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    
    x = se_block(x) 
    
    if shortcut.shape[-1] != units:
        shortcut = layers.Dense(units)(shortcut)
        
    x = layers.Add()([x, shortcut])
    x = layers.Activation('swish')(x)
    return x

def build_model(input_shape):
    inputs = layers.Input(shape=(input_shape,))
    x = layers.GaussianNoise(0.001)(inputs) # Reduced noise for precision

    x = layers.Dense(256, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)

    # Simplified layers to prevent overfitting
    x = residual_block(x, 512, 0.3)
    x = residual_block(x, 512, 0.3)
    x = residual_block(x, 256, 0.2)

    x = layers.Dense(128, activation='swish')(x)
    outputs = layers.Dense(3, activation='linear')(x)
    
    return models.Model(inputs, outputs)

model = build_model(input_dim)

# 4. TRAINING SETUP
optimizer = tf.keras.optimizers.AdamW(learning_rate=0.0005, weight_decay=0.02)
model.compile(optimizer=optimizer, loss=weighted_pose_loss, metrics=['mae'])

# Callbacks
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)
lr_reducer = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=12, min_lr=1e-6)
metrics_call = AccuracyF1Callback(validation_data=(X_test, y_test))

# 5. EXECUTE TRAINING
history = model.fit(
    X_train, y_train,
    sample_weight=train_weights,
    epochs=200,
    batch_size=256,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, lr_reducer, metrics_call],
    shuffle=True
)

model.save(os.path.join(backend_dir, "head_pose_model_2.h5"))

# 6. FINAL REPORT & GRAPHS
predictions = model.predict(X_test)
error_deg = np.abs(predictions - y_test) * (180.0 / np.pi)

print(f"\n--- Final Results ---")
print(f"Accuracy (Within 5.0°): {np.mean(np.all(error_deg < 5.0, axis=1)) * 100:.2f}%")
print(f"Pitch MAE: {np.mean(error_deg[:, 0]):.2f}°")
print(f"Yaw MAE:   {np.mean(error_deg[:, 1]):.2f}°")

# Plotting results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss (Weighted Huber)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(metrics_call.acc_history, label='Accuracy %', color='green')
plt.title('Accuracy Within 5 Degrees')
plt.ylim(0, 100)
plt.legend()
plt.show()