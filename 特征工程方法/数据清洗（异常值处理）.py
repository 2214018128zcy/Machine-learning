import pandas as pd

# 创建包含异常值的示例数据
data = {'A': [10, 15, 12, 17, 9, 30, 8, 14, 16, 20, 5]}
df = pd.DataFrame(data)

# 识别异常值的一种常见方法是使用标准差
mean = df['A'].mean()   #计算列'A'的均值（mean）
std_dev = df['A'].std() #计算列'A'的标准差（std_dev）
threshold = 2           #指定阈值（threshold），通常选择2作为标准差的倍数

'''使用布尔条件筛选出位于均值加减阈值乘以标准差范围内的数据点，这些数据点被认为是非异常值。
将筛选后的数据赋值给DataFrame df，即保留了非异常值的数据。'''
df = df[(df['A'] > mean - threshold * std_dev) & (df['A'] < mean + threshold * std_dev)]

print(df)