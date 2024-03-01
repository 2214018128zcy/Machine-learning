from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# 原始特征数据
X = np.array([
    [0.5, 0.2, 0.1, 0.8],
    [0.3, 0.6, 0.4, 0.5],
    [0.9, 0.1, 0.6, 0.3],
    [0.2, 0.8, 0.9, 0.1],
    [0.7, 0.4, 0.2, 0.6]
])

# 目标变量
y = np.array([0, 1, 1, 0, 1])

# 创建随机森林分类器作为评估器
estimator = RandomForestClassifier()

# 创建RFE对象，选择两个特征
selector = RFE(estimator, n_features_to_select=2)

# 使用fit_transform进行特征选择
selected_features = selector.fit_transform(X, y)

# 打印经过特征选择后的特征子集
print("特征选择后的特征子集：")
print(selected_features)