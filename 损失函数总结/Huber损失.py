import numpy as np
import matplotlib.pyplot as plt

def huber_loss(y, f, delta):
    absolute_errors = np.abs(y - f)
    quadratic_errors = 0.5 * np.square(y - f)
    mask = absolute_errors <= delta
    loss = np.where(mask, quadratic_errors, delta * absolute_errors - 0.5 * delta**2)
    return loss

# 生成一些样本数据
np.random.seed(1)
x = np.linspace(-5, 5, 100)
y = 2*x + 1 + np.random.randn(100) * 2  # 添加噪声

# 预测函数
def predict(x, w, b):
    return w * x + b

# 计算Huber损失
w = 0.5
b = 0.5
delta = 1.0
loss = huber_loss(y, predict(x, w, b), delta)
average_loss = np.mean(loss)

# 绘制数据点和拟合线
plt.scatter(x, y, s=10, label='Data')
plt.plot(x, predict(x, w, b), color='red', label='Fit')
plt.title('Huber Loss')
plt.xlabel('x')
plt.ylabel('y')
plt.text(-6, 15, 'Average Huber Loss = {:.2f}'.format(average_loss), ha='left', va='center')
plt.legend()
plt.show()