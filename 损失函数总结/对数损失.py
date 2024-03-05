from sklearn.metrics import log_loss
import numpy as np

# 真实标签
y_true = np.array([0, 1, 1, 0])

# 模型的预测概率
y_pred = np.array([0.2, 0.7, 0.9, 0.3])

# 计算对数损失
loss = log_loss(y_true, y_pred)

print("对数损失:", loss)