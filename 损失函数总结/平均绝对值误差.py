import numpy as np
import matplotlib.pyplot as plt

# 示例数据：真实值和预测值
true_values = np.array([10, 20, 30, 40, 50])
predicted_values = np.array([12, 18, 28, 35, 45])

# 计算平均绝对值误差（MAE）
mae = np.mean(np.abs(predicted_values - true_values))

# 绘制折线图
plt.plot(range(len(true_values)), true_values, label='True Values')
plt.plot(range(len(predicted_values)), predicted_values, label='Predicted Values')

# 添加图例
plt.legend()

# 设置坐标轴标签
plt.xlabel('Sample')
plt.ylabel('Value')

# 设置标题
plt.title('True Values vs Predicted Values')

# 显示均值绝对值误差（MAE）
plt.text(2, 40, f'MAE: {mae:.2f}', fontsize=12)

# 显示图像
plt.show()

# 输出结果
print(f"平均绝对值误差（MAE）：{mae}")