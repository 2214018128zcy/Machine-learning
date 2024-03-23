from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
import graphviz

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建并拟合C4.5决策树模型
decision_tree = tree.DecisionTreeClassifier(criterion='entropy')
decision_tree.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = decision_tree.predict(X_test)

# 计算准确率
accuracy = metrics.accuracy_score(y_test, y_pred)
print("准确率:", accuracy)

# 输出决策树结果
feature_names = iris.feature_names
class_names = iris.target_names
dot_data = tree.export_graphviz(decision_tree, out_file=None, feature_names=feature_names, class_names=class_names)
graph = graphviz.Source(dot_data)
graph.render('decision_tree')  # 保存决策树图像为'decision_tree.pdf'