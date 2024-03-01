from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# 原始特征数据
X = np.array([
    [1, 1, 1, 0],
    [0, 1, 1, 0],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [1, 1, 1, 1]
])

# 目标变量
y = np.array([1, 0, 0, 1, 1])

# 创建决策树分类器作为评估器
estimator = DecisionTreeClassifier()

# 创建SelectFromModel对象，基于特征重要性选择特征
selector = SelectFromModel(estimator)

# 使用fit_transform进行特征选择
selected_features = selector.fit_transform(X, y)

# 打印经过特征选择后的特征子集
print("特征选择后的特征子集：")
print(selected_features)