import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
features = 20
neurons = 5
x = tf.placeholder(tf.float32,(None,features))
W = tf.Variable(tf.random_normal((features,neurons)))
b = tf.Variable(tf.zeros((neurons)))
xW = tf.matmul(x,W)
z = tf.add(xW , b)
a = tf.sigmoid(z)
with tf.Session() as sess:
    init  = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(a,feed_dict={x:np.random.random([1,features])}))
print("\n Simple Regression\n")

x_data = np.linspace(10,100,10) + np.random.uniform(-10,10,10)
y_label = np.linspace(25,50,10) + np.random.uniform(-10,10,10)
#plt.plot(x_data,y_label,'*')
plt.show()
#y = mx + c // Wx + b
x = tf.placeholder(tf.float32)
w = tf.Variable(0.33)
b = tf.Variable(0.44)
error = 0
for x,y in zip(x_data,y_label):
    y_cal = w*x + b
    
    error += (y-y_cal)**2
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
train = optimizer.minimize(error)
with tf.Session() as sess:
    init  = tf.global_variables_initializer()
    sess.run(init)
    training_steps =10
    for i in range(training_steps):
        sess.run(train)
    f_w , f_b = sess.run([w,b])
x_test = np.linspace(-5,50,10)
y_p = f_w*x_test + f_b
plt.plot(x_data,y_p,'r')
plt.plot(x_data,y_label,'*')
plt.show()