import numpy as np
import matplotlib.pyplot as plt

def kl_divergence(p, q):
    return np.sum(p * np.log(p / q))
import numpy as np
import matplotlib.pyplot as plt

def kl_divergence(p, q):
    p = np.maximum(p, 1e-10)  # 将小于1e-10的概率值替换为1e-10
    q = np.maximum(q, 1e-10)
    return np.sum(p * np.log(p / q))

# 生成两个离散概率分布
x = np.linspace(0.01, 1, 100)
p = np.sin(x * np.pi)
q = np.cos(x * np.pi)

# 计算KL散度
kl = kl_divergence(p, q)

# 绘制概率分布和KL散度
plt.plot(x, p, label='P(x)')
plt.plot(x, q, label='Q(x)')
plt.legend()
plt.title('Kullback-Leibler Divergence')
plt.xlabel('x')
plt.ylabel('Probability')
plt.text(0.5, 0.8, 'KL(P || Q) = {:.4f}'.format(kl), ha='center', va='center', transform=plt.gca().transAxes)
plt.show()
# 生成两个离散概率分布
x = np.linspace(0.01, 1, 100)
p = np.sin(x * np.pi)
q = np.cos(x * np.pi)

# 计算KL散度
kl = kl_divergence(p, q)

# 绘制概率分布和KL散度
plt.plot(x, p, label='P(x)')
plt.plot(x, q, label='Q(x)')
plt.legend()
plt.title('Kullback-Leibler Divergence')
plt.xlabel('x')
plt.ylabel('Probability')
plt.text(0.5, 0.8, 'KL(P || Q) = {:.4f}'.format(kl), ha='center', va='center', transform=plt.gca().transAxes)
plt.show()