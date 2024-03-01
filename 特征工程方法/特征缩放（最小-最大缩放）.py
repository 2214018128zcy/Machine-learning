import numpy as np

# 原始特征向量
X = np.array([10, 20, 30, 40, 50])

# 最小值和最大值
X_min = X.min()
X_max = X.max()

# 最小-最大缩放
X_scaled = (X - X_min) / (X_max - X_min)

# 打印缩放后的特征向量
print("缩放后的特征向量：")
print(X_scaled)