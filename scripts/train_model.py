import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt

# Configuration constants
DEG_THRESH = 5.0
HUBER_DELTA = 2.0   
MARGIN_WEIGHT = 0.7 
ACC_INTERVAL = 5    

# Paths setup
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(backend_dir, 'data')
models_dir = os.path.join(backend_dir, 'models')
log_root = os.path.join(backend_dir, "logs/fit/")

# Ensure directories exist
os.makedirs(models_dir, exist_ok=True)
os.makedirs(log_root, exist_ok=True)

log_dir = os.path.join(log_root, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# Load dataset
X_train = np.load(os.path.join(data_dir, "X_train_final.npy"))
y_train = np.load(os.path.join(data_dir, "y_train_final.npy"))
X_test  = np.load(os.path.join(data_dir, "X_test_final.npy"))
y_test  = np.load(os.path.join(data_dir, "y_test_final.npy"))

y_train_deg = y_train * (180.0 / np.pi)
y_test_deg  = y_test  * (180.0 / np.pi)

def to_sincos(y_deg):
    y_rad = np.deg2rad(y_deg)
    return np.stack([
        np.sin(y_rad[:, 0]), np.cos(y_rad[:, 0]),
        np.sin(y_rad[:, 1]), np.cos(y_rad[:, 1]),
        np.sin(y_rad[:, 2]), np.cos(y_rad[:, 2])
    ], axis=1)

y_train_sc = to_sincos(y_train_deg)
y_test_sc  = to_sincos(y_test_deg)

# Sample weights
def get_sample_weights(labels_deg):
    pitch = np.abs(labels_deg[:, 0])
    return 1.0 + 12.0 * pitch + 2.0 * np.abs(labels_deg[:, 1])

train_weights = get_sample_weights(y_train_deg)

# Loss function
def pose_loss(y_true, y_pred):
    huber = tf.keras.losses.Huber(delta=HUBER_DELTA)

    def angle_error(i):
        sin_t, cos_t = y_true[:, i], y_true[:, i+1]
        sin_p, cos_p = y_pred[:, i], y_pred[:, i+1]
        true_ang = tf.atan2(sin_t, cos_t)
        pred_ang = tf.atan2(sin_p, cos_p)
        return tf.abs((true_ang - pred_ang) * 180.0 / np.pi)

    err_p = angle_error(0)
    err_y = angle_error(2)
    err_r = angle_error(4)

    huber_loss = (
        huber(err_p, tf.zeros_like(err_p)) * 4.0 + 
        huber(err_y, tf.zeros_like(err_y)) +
        huber(err_r, tf.zeros_like(err_r))
    )

    margin_penalty = (
        tf.nn.relu(err_p - DEG_THRESH) * 2.0 + 
        tf.nn.relu(err_y - DEG_THRESH) +
        tf.nn.relu(err_r - DEG_THRESH)
    )

    return huber_loss + MARGIN_WEIGHT * tf.reduce_mean(margin_penalty)

#------------------------------
# Model Architecture
#------------------------------

# this block helps the model "pay attention" to the most important features
# it squeezes the data down and then expands it to highlight useful information
def se_block(x, ratio=8):
    filters = x.shape[-1]
    # find the most important patterns by narrowing the focus
    se = layers.Dense(max(filters // ratio, 4), activation='relu', use_bias=False)(x)
    # create a "weight" between 0 and 1 for each feature to say how much it matters
    se = layers.Dense(filters, activation='sigmoid', use_bias=False)(se)
    # multiply the original data by these weights to emphasize the best parts
    return layers.Multiply()([x, se])

# this block allows the model to learn complex patterns without getting "confused"
# it uses a shortcut (skip connection) so the model can remember original data easily
def residual_block(x, units, dropout):
    skip = x # keep a copy of the input as a "shortcut"
    
    # process the data through layers that normalize and activate the numbers
    x = layers.Dense(units, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x) # swish helps the model learn smooth mathematical curves
    x = layers.Dropout(dropout)(x) # randomly ignore some data to prevent the model from memorizing

    x = layers.Dense(units, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = se_block(x) # apply the attention block from above

    # if the shortcut size doesn't match the new data size, adjust the shortcut
    if skip.shape[-1] != units:
        skip = layers.Dense(units)(skip)
        
    # add the original shortcut back to the processed data
    # this helps training stay stable even as the model gets deeper
    x = layers.Add()([x, skip])
    return layers.Activation('swish')(x)

# the main blueprint for building the entire neural network
def build_model(input_dim):
    inp = layers.Input(shape=(input_dim,))
    
    # add a tiny bit of blur to the input so the model doesn't rely on perfect data
    # this makes it work better on real-world, slightly shaky video
    x = layers.GaussianNoise(0.005)(inp) 

    # start with a medium-sized layer to identify basic landmark relationships
    x = layers.Dense(256, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)

    # use three residual blocks to dive deep into the geometry of the face
    # the capacity grows to 512 to handle complex math, then shrinks back down
    x = residual_block(x, 512, 0.4) 
    x = residual_block(x, 512, 0.4)
    x = residual_block(x, 256, 0.3)

    # narrow the data down to the most essential information
    x = layers.Dense(128, activation='swish')(x)
    
    # output the final 6 numbers (sine and cosine for pitch, yaw, and roll)
    out = layers.Dense(6, activation='linear')(x)
    return models.Model(inp, out)

model = build_model(X_train.shape[1])

# Callbacks
class StrictAccuracyCallback(callbacks.Callback):
    def __init__(self, X_val, y_val_deg, interval=5):
        super().__init__()
        self.X = X_val
        self.y = y_val_deg
        self.interval = interval

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval != 0: return
        preds = self.model(self.X, training=False).numpy()
        p = np.arctan2(preds[:, 0], preds[:, 1])
        y = np.arctan2(preds[:, 2], preds[:, 3])
        r = np.arctan2(preds[:, 4], preds[:, 5])
        pred_deg = np.stack([p, y, r], axis=1) * 180.0 / np.pi
        err = np.abs(pred_deg - self.y)
        acc = np.mean(np.all(err < DEG_THRESH, axis=1)) * 100
        print(f" — strict acc (≤5°): {acc:.2f}%")

model.compile(
    optimizer=tf.keras.optimizers.AdamW(3e-4, weight_decay=0.04), 
    loss=pose_loss
)

callbacks_list = [
    StrictAccuracyCallback(X_test, y_test_deg, interval=ACC_INTERVAL),
    callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
    callbacks.TerminateOnNaN(), 
    callbacks.EarlyStopping(patience=35, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-7)
]

# Run training
print(f"View real-time stats: tensorboard --logdir {log_root}")

history = model.fit(
    X_train, y_train_sc,
    sample_weight=train_weights,
    validation_data=(X_test, y_test_sc),
    epochs=200,
    batch_size=256,
    callbacks=callbacks_list,
    verbose=1
)

# Save to the specific models directory
save_path = os.path.join(models_dir, "head_pose_model.h5")
model.save(save_path)
print(f"Model saved successfully to: {save_path}")