from sklearn.feature_extraction.text import TfidfVectorizer

# 原始文本数据
documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?"
]

# 创建TF-IDF向量化器
vectorizer = TfidfVectorizer()

# 将文本转换为TF-IDF表示
tfidf_matrix = vectorizer.fit_transform(documents)

# 获取TF-IDF表示的特征向量
feature_names = vectorizer.get_feature_names()

# 打印TF-IDF表示的特征向量
print(tfidf_matrix.toarray())

# 打印特征向量对应的词语
print(feature_names)