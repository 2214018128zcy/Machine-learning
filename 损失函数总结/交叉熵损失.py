
'''
import numpy as np

# 模型的预测输出
y_pred = np.array([0.2, 0.8, 0.6, 0.3])

# 真实标签
y_true = np.array([0, 1, 1, 0])

# 计算交叉熵损失
loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

print("交叉熵损失:", loss)
'''


import numpy as np

# 模型的预测输出
y_pred = np.array([[0.1, 0.4, 0.5],
                   [0.8, 0.1, 0.1],
                   [0.3, 0.6, 0.1]])

# 真实标签，one-hot编码
y_true = np.array([[0, 0, 1],
                   [1, 0, 0],
                   [0, 1, 0]])

# 计算交叉熵损失
loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

print("交叉熵损失:", loss)