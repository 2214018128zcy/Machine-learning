import nltk
from nltk.corpus import stopwords

text = "This is an example sentence, demonstrating the removal of stop words."

# 下载停用词
nltk.download('stopwords')

# 获取英语停用词列表
stop_words = set(stopwords.words('english'))

# 分词
tokens = nltk.word_tokenize(text)

# 移除停用词
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

print(filtered_tokens)