import numpy as np

# 原始数据
time = np.array([1, 2, 3, 4, 5])
storage = np.array([10, 6, 3, 2, 1])

# 指数变换
exp_storage = np.exp(storage)

print("原始货物存储量：", storage)
print("指数变换后的货物存储量：", exp_storage)