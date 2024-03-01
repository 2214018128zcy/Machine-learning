from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# 加载鸢尾花数据集
data = load_iris()
X, y = data.data, data.target

# 创建PCA对象
pca = PCA(n_components=2)

# 对数据进行降维
X_pca = pca.fit_transform(X)

print(X_pca.shape)