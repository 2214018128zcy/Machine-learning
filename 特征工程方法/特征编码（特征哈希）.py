from sklearn.feature_extraction.text import HashingVectorizer

# 原始文本数据
documents = [
    "这部电影太好看了，情节紧凑、演员表现出色！",
    "我觉得这部电影很无聊，剧情一点都不吸引人。"
]

# 创建HashingVectorizer对象
vectorizer = HashingVectorizer(n_features=10)

# 使用transform进行特征哈希编码
hashed_features = vectorizer.transform(documents)

# 打印特征哈希编码后的结果
print("特征哈希编码后的结果：")
print(hashed_features.toarray())