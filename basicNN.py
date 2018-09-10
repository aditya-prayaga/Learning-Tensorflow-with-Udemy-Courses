import tensorflow as tf 
import numpy as np 

np.random.seed(101)
tf.set_random_seed(101)


a = np.random.uniform(0,100,(5,5))
print(a)
print("\n")
b = np.random.uniform(0,100,(5,1))
print(b)

pha = tf.placeholder(tf.float32)
phb = tf.placeholder(tf.float32) 

add_op = pha + phb 
mul = pha * phb


with tf.Session() as sess:
    print(sess.run(add_op,feed_dict={pha:a,phb:b}))
    print("\n")
    print(sess.run(mul,feed_dict={pha:a,phb:b}))