import pandas as pd
import statsmodels.api as sm

# 创建示例时间序列数据
data = pd.DataFrame({
    '日期': ['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04', '2022-01-05'],
    '气温': [15, 18, 20, 16, 22]
})

# 将日期列转换为日期时间类型
data['日期'] = pd.to_datetime(data['日期'])

# 提取趋势特征
X_trend = sm.add_constant(range(len(data)))
model_trend = sm.OLS(data['气温'], X_trend)
results_trend = model_trend.fit()
trend_slope = results_trend.params[1]  # 趋势斜率

# 提取周期性特征
X_period = sm.add_constant(range(len(data)))
model_period = sm.OLS(data['气温'], X_period)
results_period = model_period.fit()
period_frequency = 1 / results_period.params[2]  # 周期频率

# 提取季节性特征
seasonal_decomposition = sm.tsa.seasonal_decompose(data['气温'], period=1)
seasonal_index = seasonal_decomposition.seasonal

# 提取自回归特征
lag_values = [1, 2, 3]  # 指定滞后观测值的数量
lag_features = []
for lag in lag_values:
    lag_feature = data['气温'].shift(lag)
    lag_features.append(lag_feature)
data['lag_1'] = lag_features[0]
data['lag_2'] = lag_features[1]
data['lag_3'] = lag_features[2]

# 提取统计特征
mean_temperature = data['气温'].mean()
var_temperature = data['气温'].var()
max_temperature = data['气温'].max()
min_temperature = data['气温'].min()

# 打印结果
print("趋势斜率:", trend_slope)
print("周期频率:", period_frequency)
print("季节性指数:", seasonal_index)
print("滞后观测值:", lag_features)
print("均值:", mean_temperature)
print("方差:", var_temperature)
print("最大值:", max_temperature)
print("最小值:", min_temperature)