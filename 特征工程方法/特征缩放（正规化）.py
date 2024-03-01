from sklearn.preprocessing import Normalizer
import numpy as np

# 原始特征向量
X = np.array([[3, 4]])

# 创建Normalizer对象
normalizer = Normalizer(norm='l2')

# 使用transform进行正规化
X_normalized = normalizer.transform(X)

# 打印正规化后的特征向量
print("正规化后的特征向量：")
print(X_normalized)