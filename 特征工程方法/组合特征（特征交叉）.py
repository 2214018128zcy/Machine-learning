import pandas as pd

# 示例数据集
data = {'feature1': [3, 4, 5, 6],
        'feature2': [6, 7, 8, 9]}
df = pd.DataFrame(data)

# 特征交叉
df['feature_cross'] = df['feature1'] * df['feature2']

print(df)