import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定字体为宋体

# 定义真实值和预测值
y_true = [2, 4, 6, 8, 10]
y_pred = [1.5, 3.5, 5.5, 7.5, 9.5]

# 将列表转换为NumPy数组
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# 计算均方根误差
rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))

# 打印结果
print("均方根误差：", rmse)

# 绘制散点图
plt.scatter(y_true, y_pred)
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')  # 绘制对角线
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('真实值 vs 预测值')
plt.text(4, 8, f'RMSE: {rmse:.2f}', fontsize=12)  # 在图中添加均方根误差值
plt.show()