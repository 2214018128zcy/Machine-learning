import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 将数据集转换为XGBoost的DMatrix格式
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 定义XGBoost参数
params = {
    'objective': 'multi:softmax',
    'num_class': 3,
    'max_depth': 3,
    'eta': 0.1,
    'seed': 42
}

# 训练XGBoost分类器
num_rounds = 100
model = xgb.train(params, dtrain, num_rounds)

# 在测试集上进行预测
y_pred = model.predict(dtest)

# 将预测结果转换为整数类型
y_pred = y_pred.astype(int)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("准确率:", accuracy)