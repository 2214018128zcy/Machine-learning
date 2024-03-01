from sklearn.preprocessing import LabelEncoder
import numpy as np

# 原始分类变量
categories = np.array(["苹果", "香蕉", "橙子", "苹果", "葡萄"])

# 创建LabelEncoder对象
encoder = LabelEncoder()

# 使用fit_transform进行标签编码
encoded_categories = encoder.fit_transform(categories)

# 打印标签编码后的结果
print("标签编码后的结果：")
print(encoded_categories)
