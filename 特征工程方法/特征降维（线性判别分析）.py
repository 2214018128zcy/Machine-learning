from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris

# 加载鸢尾花数据集
data = load_iris()
X, y = data.data, data.target
# 创建LDA对象
lda = LinearDiscriminantAnalysis(n_components=2)

# 对数据进行降维
X_lda = lda.fit_transform(X, y)

print(X_lda.shape)