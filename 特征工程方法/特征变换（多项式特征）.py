from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 原始特征数据
X = [[1500, 10], [2000, 5], [1200, 15]]
y = [250, 350, 200]

# 创建PolynomialFeatures对象，设置度数为2
poly = PolynomialFeatures(degree=2)

# 使用fit_transform进行多项式特征变换
X_poly = poly.fit_transform(X)

# 创建线性回归模型
model = LinearRegression()

# 使用多项式特征变换后的特征矩阵进行模型训练
model.fit(X_poly, y)

# 进行预测
new_data = [[1800, 8]]
new_data_poly = poly.transform(new_data)
predicted_price = model.predict(new_data_poly)

# 打印预测结果
print("预测的销售价格：", predicted_price)