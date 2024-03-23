import numpy as np
from collections import Counter

def entropy(labels):
    """计算熵"""
    counter = Counter(labels)
    probs = [counter[c] / len(labels) for c in set(labels)]
    return -np.sum(probs * np.log2(probs))

def information_gain(data, feature_index, target_index):
    """计算信息增益"""
    total_entropy = entropy(data[:, target_index])
    feature_values = set(data[:, feature_index])
    weighted_entropy = 0
    for value in feature_values:
        subset = data[data[:, feature_index] == value]
        subset_entropy = entropy(subset[:, target_index])
        weighted_entropy += len(subset) / len(data) * subset_entropy
    return total_entropy - weighted_entropy

def id3(data, target_index, features):
    """ID3算法"""
    labels = data[:, target_index]
    if len(set(labels)) == 1:
        return labels[0]
    if len(features) == 0:
        return Counter(labels).most_common(1)[0][0]
    best_feature_index = max(features, key=lambda index: information_gain(data, index, target_index))
    best_feature = data[0, best_feature_index]
    tree = {best_feature: {}}
    feature_values = set(data[:, best_feature_index])
    remaining_features = [f for f in features if f != best_feature_index]  # 更新特征列表
    for value in feature_values:
        subset = data[data[:, best_feature_index] == value]
        if len(subset) == 0:
            tree[best_feature][value] = Counter(labels).most_common(1)[0][0]
        else:
            tree[best_feature][value] = id3(subset, target_index, remaining_features)
    return tree

# 示例用法
data = np.array([
    [1, 'Sunny', 'Hot', 'High', 'Weak', 'No'],
    [2, 'Sunny', 'Hot', 'High', 'Strong', 'No'],
    [3, 'Overcast', 'Hot', 'High', 'Weak', 'Yes'],
    [4, 'Rain', 'Mild', 'High', 'Weak', 'Yes'],
    [5, 'Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
    [6, 'Rain', 'Cool', 'Normal', 'Strong', 'No'],
    [7, 'Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
    [8, 'Sunny', 'Mild', 'High', 'Weak', 'No'],
    [9, 'Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
    [10, 'Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
    [11, 'Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
    [12, 'Overcast', 'Mild', 'High', 'Strong', 'Yes'],
    [13, 'Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
    [14, 'Rain', 'Mild', 'High', 'Strong', 'No']
])

target_index = -1
features = list(range(1, data.shape[1] - 1))  # 特征列索引

tree = id3(data, target_index, features)
print(tree)

#输出结果
'''
{'Outlook': {'Sunny': {'Humidity': {'High': 'No', 'Normal': 'Yes'}},
             'Overcast': 'Yes',
             'Rain': {'Wind': {'Weak': 'Yes', 'Strong': 'No'}}}}
'''