from sklearn.feature_selection import VarianceThreshold
import numpy as np

# 原始特征数据
X = np.array([
    [1, 1, 1, 0],
    [0, 1, 1, 0],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [1, 1, 1, 1]
])

# 创建VarianceThreshold对象
selector = VarianceThreshold(threshold=0.2)

# 使用fit_transform进行特征选择
selected_features = selector.fit_transform(X)

# 打印经过方差选择后的特征子集
print("方差选择后的特征子集：")
print(selected_features)