import pandas as pd

# 创建示例时间序列数据
data = pd.DataFrame({
    '日期': ['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04', '2022-01-05'],
    '销售额': [100, 120, 90, 110, 130]
})

# 将日期列转换为日期时间类型
data['日期'] = pd.to_datetime(data['日期'])

# 计算简单移动平均特征
window_size = 7  # 窗口大小为7天
data['moving_avg'] = data['销售额'].rolling(window=window_size).mean()

# 打印结果
print(data)