import tensorflow as tf 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Actual equation y = 5*x + 15 + noise
x_data  = np.linspace(0,10,10000)
noise = np.random.randn(len(x_data))
y_real = 5*x_data + 15 + noise
x_df = pd.DataFrame(x_data,columns = ["X Data"])
y_df = pd.DataFrame(y_real,columns = ["Y"])
my_d = pd.concat([x_df,y_df],axis=1)

#my_d.sample(n=250).plot(kind = 'scatter',x ="X Data",y = 'Y' )
#plt.show()
batch_len = 10

w = tf.Variable(0.001)
b = tf.Variable(0.111)
xph = tf.placeholder(tf.float32,[batch_len]) 
yph = tf.placeholder(tf.float32,[batch_len])
y_lab = w*xph + b
error = tf.reduce_sum(tf.square((yph-y_lab)))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
train = optimizer.minimize(error)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    batches = 1000
    for i in range(batches):
        ind = np.random.randint(len(x_data),size =batch_len)
        feed = {xph:x_data[ind],yph:y_real[ind]}
        sess.run(train,feed_dict=feed) 
    m_m ,m_b = sess.run([w,b])
    print(m_m,m_b)

    