import numpy as np

# 原始特征向量
X = np.array([10, 20, 30, 40, 50])

# 计算均值和标准差
mean = np.mean(X)
std_dev = np.std(X)

# 标准化
X_scaled = (X - mean) / std_dev

# 打印标准化后的特征向量
print("标准化后的特征向量：")
print(X_scaled)