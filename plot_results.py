import os
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 基础路径设置
# =========================
RESULT_DIR = './results'
FIG_DIR = './figures'

os.makedirs(FIG_DIR, exist_ok=True)

# =========================
# 文件检查
# =========================
required_files = [
    'train_loss.npy',
    'vali_loss.npy',
    'preds.npy',
    'trues.npy'
]

for f in required_files:
    path = os.path.join(RESULT_DIR, f)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")

# =========================
# 加载数据
# =========================
train_loss = np.load(os.path.join(RESULT_DIR, 'train_loss.npy'))
vali_loss  = np.load(os.path.join(RESULT_DIR, 'vali_loss.npy'))
preds = np.load(os.path.join(RESULT_DIR, 'preds.npy'))
trues = np.load(os.path.join(RESULT_DIR, 'trues.npy'))

# =========================
# 可视化参数选择
# =========================
sample_index = 0   # 第几个样本
variable_index = 0 # 第几个变量（多变量预测时）

# =========================
# 1. Loss 收敛曲线
# =========================
plt.figure(figsize=(6, 4))
plt.plot(train_loss, label='Training Loss', linewidth=2)
plt.plot(vali_loss, label='Validation Loss', linewidth=2)

plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Training and Validation Loss Curves')

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'loss_curve.png'), dpi=300)
plt.close()

# =========================
# 2. Prediction vs Ground Truth
# =========================
plt.figure(figsize=(8, 4))
plt.plot(
    trues[sample_index, :, variable_index],
    label='Ground Truth',
    linewidth=2
)
plt.plot(
    preds[sample_index, :, variable_index],
    label='Prediction',
    linestyle='--'
)

plt.xlabel('Time Step')
plt.ylabel('Target Value')
plt.title('Prediction versus Ground Truth')

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'prediction_vs_truth.png'), dpi=300)
plt.close()

# =========================
# 3. Absolute Error Curve
# =========================
absolute_error = np.abs(
    preds[sample_index, :, variable_index] -
    trues[sample_index, :, variable_index]
)

plt.figure(figsize=(8, 4))
plt.plot(absolute_error, color='tab:red')

plt.xlabel('Time Step')
plt.ylabel('Absolute Error')
plt.title('Absolute Prediction Error over Time')

plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'absolute_error_curve.png'), dpi=300)
plt.close()

# =========================
# 控制台提示
# =========================
print('========================================')
print(' All figures have been successfully saved')
print(f' Figure directory: {os.path.abspath(FIG_DIR)}')
print('========================================')
