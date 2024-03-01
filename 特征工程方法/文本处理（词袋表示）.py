from sklearn.feature_extraction.text import CountVectorizer

# 原始文本数据
documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?"
]

# 创建词袋模型
vectorizer = CountVectorizer()

# 将文本转换为词袋表示
bag_of_words = vectorizer.fit_transform(documents)

# 获取词袋中的词语
feature_names = vectorizer.get_feature_names()

# 打印词袋表示的特征向量
print(bag_of_words.toarray())

# 打印词袋中的词语
print(feature_names)