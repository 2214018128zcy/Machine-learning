import tensorflow as tf

# 两个商品的特征向量
x = tf.constant([0.5, 0.8, 0.2], dtype=tf.float32)
y = tf.constant([0.7, 0.3, 0.9], dtype=tf.float32)

# 计算余弦相似度
cos_similarity = tf.reduce_sum(x * y) / (tf.norm(x) * tf.norm(y))
cos_similarity_loss = 1 - cos_similarity

print("余弦相似度损失:", cos_similarity_loss.numpy())