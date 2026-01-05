import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# Config
deg_thresh_strict = 5.0
deg_thresh_robust = 10.0

# Paths setup
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(base_dir, 'data')
models_dir = os.path.join(base_dir, 'models')
assets_dir = os.path.join(base_dir, 'assets')
model_path = os.path.join(models_dir, "head_pose_model.h5")

os.makedirs(assets_dir, exist_ok=True)

# Load data and model
print("loading data and model...")
x_test = np.load(os.path.join(data_dir, "x_test_final.npy"))
y_test_rad = np.load(os.path.join(data_dir, "y_test_final.npy"))
y_test_deg = y_test_rad * (180.0 / np.pi)

if not os.path.exists(model_path):
    raise FileNotFoundError(f"model not found at {model_path}")

model = tf.keras.models.load_model(model_path, compile=False)

# Inference
print("running inference...")
preds_sc = model.predict(x_test, verbose=0)

# Convert sine-cosine back to degrees
def from_sincos(sc):
    p = np.arctan2(sc[:, 0], sc[:, 1])
    y = np.arctan2(sc[:, 2], sc[:, 3])
    r = np.arctan2(sc[:, 4], sc[:, 5])
    return np.stack([p, y, r], axis=1) * (180.0 / np.pi)

preds_deg = from_sincos(preds_sc)
errors = np.abs(preds_deg - y_test_deg)

# Metric calculations
mae = np.mean(errors, axis=0)
acc_5 = np.mean(np.all(errors < deg_thresh_strict, axis=1)) * 100
acc_10 = np.mean(np.all(errors < deg_thresh_robust, axis=1)) * 100

# Visualization setup
plt.style.use('dark_background')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f"loomis-lens performance analysis", fontsize=22, color='#00ffcc', fontweight='bold')

# Plot error probability (cdf)
for i, (label, color) in enumerate(zip(['pitch', 'yaw', 'roll'], ['#ff3366', '#3399ff', '#00ffcc'])):
    sorted_err = np.sort(errors[:, i])
    cdf = np.arange(len(sorted_err)) / float(len(sorted_err))
    axes[0,0].plot(sorted_err, cdf, label=f"{label} (mae: {mae[i]:.2f}°)", color=color, linewidth=3)

axes[0,0].axvspan(0, 5, color='green', alpha=0.1, label='goal (≤5°)')
axes[0,0].set_xlim(0, 15)
axes[0,0].set_title("success probability (cdf)", fontsize=16)
axes[0,0].legend(loc='lower right')
axes[0,0].grid(alpha=0.1)

# Plot pitch error across angles
axes[0,1].scatter(y_test_deg[:, 0], errors[:, 0], alpha=0.15, color='cyan', s=3)
sns.regplot(x=y_test_deg[:, 0], y=errors[:, 0], scatter=False, ax=axes[0,1], color='#ff3366')
axes[0,1].set_title("pitch robustness", fontsize=16)
axes[0,1].set_ylim(0, 15)
axes[0,1].grid(alpha=0.1)

# Plot error distribution per axis
sns.boxplot(data=errors, ax=axes[1,0], palette="cool", width=0.4, fliersize=1)
axes[1,0].set_xticklabels(['pitch', 'yaw', 'roll'])
axes[1,0].set_title("error distribution", fontsize=16)
axes[1,0].set_ylim(0, 12)
axes[1,0].grid(axis='y', alpha=0.1)

# Plot reliability pie chart
bins = ['high precision', 'robust', 'outlier']
success_5 = np.all(errors < deg_thresh_strict, axis=1)
success_10 = np.all(errors < deg_thresh_robust, axis=1)
counts = [np.sum(success_5), np.sum(success_10 & ~success_5), np.sum(~success_10)]

axes[1,1].pie(counts, labels=bins, autopct='%1.1f%%', colors=['#00ffcc', '#3399ff', '#ff3366'], startangle=140)
axes[1,1].set_title("prediction reliability", fontsize=16)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save and show
output_plot = os.path.join(assets_dir, "portfolio_metrics.png")
plt.savefig(output_plot, dpi=300)
plt.show()

print(f"overall mae: {np.mean(mae):.2f}°")
print(f"accuracy: {acc_5:.2f}%")