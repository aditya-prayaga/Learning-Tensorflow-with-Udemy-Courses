import tensorflow as tf
print(tf.get_default_graph())
g1 = tf.Graph()
print(g1)
g2 = tf.Graph()
print(g2)
with g2.as_default():
    print(tf.get_default_graph())

###########VALUES PLACEHOLDERS###########


ex = tf.random_normal((4,4),0,1)
mvar = tf.Variable(initial_value=ex)
print(mvar)
#sess.run(mvar)
sess = tf.InteractiveSession() 
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(mvar))
ph = tf.placeholder(tf.float32,(None,5))