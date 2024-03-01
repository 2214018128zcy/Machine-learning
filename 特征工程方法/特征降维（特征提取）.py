from sklearn.decomposition import KernelPCA
from sklearn.datasets import load_iris

# 加载鸢尾花数据集
data = load_iris()
X, y = data.data, data.target
# 创建KernelPCA对象
kpca = KernelPCA(n_components=2, kernel='rbf')

# 对数据进行特征提取
X_kpca = kpca.fit_transform(X)

print(X_kpca.shape)