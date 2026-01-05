import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('./figures', exist_ok=True)

# 1. Loss 曲线
train = np.load('./results/train_loss.npy')
vali = np.load('./results/vali_loss.npy')

plt.figure(figsize=(6,4))
plt.plot(train, label='Train')
plt.plot(vali, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.tight_layout()
plt.savefig('./figures/loss_curve.png', dpi=300)
plt.close()

# 2. 预测 vs 真实
preds = np.load('./results/preds.npy')
trues = np.load('./results/trues.npy')

idx, var = 0, 0

plt.figure(figsize=(8,4))
plt.plot(trues[idx,:,var], label='Ground Truth')
plt.plot(preds[idx,:,var], label='Prediction')
plt.legend()
plt.tight_layout()
plt.savefig('./figures/pred_vs_true.png', dpi=300)
plt.close()

# 3. 误差曲线
error = np.abs(preds - trues)

plt.figure(figsize=(8,4))
plt.plot(error[idx,:,var])
plt.xlabel('Time step')
plt.ylabel('Absolute Error')
plt.tight_layout()
plt.savefig('./figures/error_curve.png', dpi=300)
plt.close()

print("Figures saved to ./figures/")
