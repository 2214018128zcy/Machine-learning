from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# 加载California housing数据集
housing = fetch_california_housing()
X = housing.data
y = housing.target

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建并拟合MARS模型
mars_model = make_pipeline(PolynomialFeatures(include_bias=False), LinearRegression())
mars_model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = mars_model.predict(X_test)

# 计算均方误差
mse = mean_squared_error(y_test, y_pred)
print("均方误差:", mse)