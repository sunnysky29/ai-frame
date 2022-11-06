# -*- coding: utf-8 -*-
"""
==============================================================================
Time : 2022/11/6 10:26
File : mnist_st_demo.py
mnist 训练demo
使用 单层感知机softmax

参考：https://github.com/PnYuan/Practice-of-Machine-Learning/blob/master/code/MNIST_tensorflow_demo/softmax_demo.py
==============================================================================
"""

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data  # for data loading
import matplotlib.pyplot as plt  # for image showing


#========== data loading ==========#
mnist = input_data.read_data_sets('../../../corpus/MNIST',one_hot=True)
X_train_org, Y_train_org = mnist.train.images, mnist.train.labels
X_valid_org, Y_valid_org = mnist.validation.images, mnist.validation.labels
X_test_org,  Y_test_org = mnist.test.images, mnist.test.labels

# check the shape of dataset
print("train set shape: X-", X_train_org.shape, ", Y-", Y_train_org.shape)
print("valid set shape: X-", X_valid_org.shape, ", Y-", Y_valid_org.shape)
print("test set shape: X-", X_test_org.shape, ", Y-", Y_test_org.shape)

n = 5
# ========== Softmax Modeling ==========#
# placeholder 提供占位 OP
x = tf.placeholder("float", [None, 28,28,1])  # placeholder of input
x = tf.reshape(x, [-1, 784])  # -1 表示batch大小，运行时自动推演

t = tf.placeholder("float", [None, 10])  # placeholder of label, 真实标签

W = tf.Variable(tf.zeros([784, 10]))  # parameters (initial to 0)
b = tf.Variable(tf.zeros([10]))

init = tf.global_variables_initializer() #  初始化

# 模型定义
y = tf.nn.softmax(tf.matmul(x, W) + b)  # softmax computation graph


# 损失函数
# loss (cross-entropy)
# here we use clip_by_value() to restrict the value between 1e-8 and 1 to avoid log(0)
cross_entropy = -tf.reduce_sum(t * tf.log(tf.clip_by_value(y, 1e-8, 1.0)))

# 优化算法
optimizer =tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_step = optimizer.minimize(cross_entropy)  # using GD

# 精度
# x--> y, 根据输入数据x ,推理得到预测 y
# y_: 真实值
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))  # use argmax() for decoding one-hot
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # training evaluation

# ========== Training ==========#
sess = tf.InteractiveSession()  # initial a session
sess.run(init)

for step in range(1000):  # iterate  for 100 times
    batch_xs, batch_ys = mnist.train.next_batch(100)  # using mini-batch
    sess.run(train_step, feed_dict={x: batch_xs, t: batch_ys})
    if step % 100 ==0:
        # ========== Evaluation ==========#
        # test on valid set / test set
        print("iter: ",step,  "train accuracy", sess.run(accuracy, feed_dict={x: batch_xs, t: batch_ys}),
              "valid accuracy", sess.run(accuracy, feed_dict={x: X_valid_org, t: Y_valid_org}),
              "test accuracy", sess.run(accuracy, feed_dict={x: X_test_org, t: Y_test_org}))

