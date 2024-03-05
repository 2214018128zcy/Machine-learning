'''
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定字体为宋体

# 模型的预测输出
y_pred = np.array([0.2, 0.8, 0.6, 0.3])

# 真实标签
y_true = np.array([0, 1, 1, 0])

# 计算交叉熵损失
loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

print("交叉熵损失:", loss)

# 绘制交叉熵损失图像
probabilities = np.linspace(0.01, 0.99, 100)  # 在0.01到0.99之间生成100个预测概率值
losses = []

for p in probabilities:
    loss = -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
    losses.append(loss)

plt.plot(probabilities, losses)
plt.xlabel('预测概率')
plt.ylabel('交叉熵损失')
plt.title('交叉熵损失与预测概率之间的关系')
plt.show()

'''
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定字体为宋体

# 模型的预测输出
y_pred = np.array([[0.1, 0.4, 0.5],
                   [0.8, 0.1, 0.1],
                   [0.3, 0.6, 0.1]])

# 真实标签，one-hot编码
y_true = np.array([[0, 0, 1],
                   [1, 0, 0],
                   [0, 1, 0]])

# 计算交叉熵损失
epsilon = 1e-10
y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # 将预测概率限制在epsilon和1-epsilon之间
loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

print("交叉熵损失:", loss)

# 绘制交叉熵损失图像
probabilities = np.linspace(0.01, 0.99, 100)  # 在0.01到0.99之间生成100个预测概率值
losses = []

for p in probabilities:
    y_pred_temp = np.array([[1 - p, p, epsilon]])
    y_pred_temp = np.clip(y_pred_temp, epsilon, 1 - epsilon)  # 将预测概率限制在epsilon和1-epsilon之间
    loss = -np.mean(np.sum(y_true * np.log(y_pred_temp), axis=1))
    losses.append(loss)

plt.plot(probabilities, losses)
plt.xlabel('预测概率')
plt.ylabel('交叉熵损失')
plt.title('交叉熵损失与预测概率之间的关系')
plt.show()