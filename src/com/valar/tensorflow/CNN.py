# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 15:19:26 2017

@author: HUAL
"""

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf

#==============构建tensorflow图=========================
#运行TensorFlow的InteractiveSession
sess = tf.InteractiveSession()

#输入图片x 是一个2维的浮点数张量 (placeholder 的shape 参数是可选的，但有了它，TensorFlow能够自动捕捉因数据维度不一致导致的错误。)
x = tf.placeholder("float", shape=[None, 784])
#一个10维的one-hot向量,用于代表对应某一MNIST图片的类别。
y_ = tf.placeholder("float", shape=[None, 10])

#---------函数定义------------
#权重初始化
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

#用一个较小的正数来初始化偏置项，以避免神经元节点输出恒为0的问题（dead neurons）。
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#卷积使用1步长（stride size），0边距（padding size）的模板，保证输入和输出一样大小
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#池化用简单传统的2x2大小的模板做max pooling。
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

#---------第一层------------

#前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道数目(定为32，即用32个不同卷积核来分别乘)。
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32]) #对于每一个输出通道都有一个对应的偏置量。

#把x 变成一个4d向量，其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数(因
#为是灰度图所以这里的通道数为1，如果是rgb彩色图，则为3)。
x_image = tf.reshape(x, [-1,28,28,1])

#我们把x_image 和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max pooling。
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#---------第二层------------

#第二层中，每个5x5的patch会得到64个特征（32个输入，64个输出，每个通道配2个卷积核）。
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#---------密集连接层------------

#现在，图片尺寸减小到7x7，我们加入一个有1024个神经元的全连接层，用于处理整个图片。我们把池化层输出的
#张量reshape成一些向量，乘上权重矩阵，加上偏置，然后对其使用ReLU。
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#---------密集连接层------------

#为了减少过拟合，我们在输出层之前加入dropout。我们用一个placeholder 来代表一个神经元的输出在dropout
#中保持不变的概率。这样我们可以在训练过程中启用dropout，在测试过程中关闭dropout。 TensorFlow的tf.nn.dropout
#操作除了可以屏蔽神经元的输出外，还会自动处理神经元输出值的scale。所以用dropout的时候可以不用考虑scale。
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#---------输出层------------

#最后，我们添加一个softmax层，就像前面的单层softmax regression一样。
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#---------训练和评估模型------------
#为了训练和评估，我们使用与之前简单的单层SoftMax神经网络模型几乎相同的一套代码，只是我们会用更加
#复杂的ADAM优化器来做梯度最速下降，在feed_dict 中加入额外的参数keep_prob 来控制dropout比例。然后每100次迭代输出一次日志。
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#==============启动并运行tensorflow代码=========================
#初始化tf变量
sess.run(tf.global_variables_initializer())

#迭代循环：玄幻20k次，每次取随机50个手写图
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5}) #dropout用0.5一般会比较好
print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
sess.close()