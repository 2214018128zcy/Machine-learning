from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建并拟合CART分类树模型
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = decision_tree.predict(X_test)

# 计算准确率
accuracy = metrics.accuracy_score(y_test, y_pred)
print("准确率:", accuracy)