import pandas as pd

# 创建包含重复行的示例数据
data = {'A': [1, 2, 3, 4, 2, 3, 1],
        'B': ['a', 'b', 'c', 'd', 'b', 'c', 'a']}
df = pd.DataFrame(data)

# 打印原始数据
print("原始数据：")
print(df)

# 删除重复行
df = df.drop_duplicates()

# 打印删除重复行后的数据
print("\n删除重复行后的数据：")
print(df)
