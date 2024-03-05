import tensorflow as tf

# 两个向量表示
x = tf.constant([1, 2, 3], dtype=tf.float32)
y = tf.constant([4, 5, 6], dtype=tf.float32)

# 计算余弦相似度
cos_similarity = tf.reduce_sum(x * y) / (tf.norm(x) * tf.norm(y))
cos_similarity_loss = 1 - cos_similarity

print("余弦相似度损失:", cos_similarity_loss.numpy())