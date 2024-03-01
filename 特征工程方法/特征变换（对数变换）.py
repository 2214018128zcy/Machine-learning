import numpy as np

# 原始数据
area = np.array([1000, 1500, 2000, 2500, 3000])
price = np.array([50, 75, 100, 125, 150])

# 对数变换
log_price = np.log10(price)

print("原始价格：", price)
print("对数变换后的价格：", log_price)
