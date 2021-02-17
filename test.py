import tensorflow as tf
import numpy as np

x = tf.constant([[[3, 8], [4, 10], [1, 15]], [[2, 7], [3, 9], [2, 6]], [[2, 4], [1, 1], [6, 8]]])
x = x[tf.newaxis, :, :, :]
print(x.shape)

y = tf.layers.max_pooling2d(x, 3, 1, padding="same")
keep = tf.cast(tf.equal(x, y), tf.int32)
z = keep * x
print(z.eval(session=tf.compat.v1.Session()))