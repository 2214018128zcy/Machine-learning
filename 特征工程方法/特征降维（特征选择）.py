from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.datasets import load_iris

# 加载鸢尾花数据集
data = load_iris()
X, y = data.data, data.target
# 创建SelectKBest对象
selector = SelectKBest(score_func=f_classif, k=2)

# 对数据进行特征选择
X_selected = selector.fit_transform(X, y)

print(X_selected.shape)