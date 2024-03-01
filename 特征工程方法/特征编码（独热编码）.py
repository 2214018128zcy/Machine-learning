from sklearn.preprocessing import OneHotEncoder
import numpy as np

# 原始分类变量
categories = np.array(["red", "blue", "green", "red", "yellow"])

# 创建OneHotEncoder对象
encoder = OneHotEncoder()

# 使用fit_transform进行独热编码
encoded_categories = encoder.fit_transform(categories.reshape(-1, 1))

# 打印独热编码后的结果
print("独热编码后的结果：")
print(encoded_categories.toarray())