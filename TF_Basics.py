# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 18:27:23 2017

@author: vsnalam
"""

from __future__ import print_function
import os
os.chdir("C:\Sreedhar\Software\Anaconda Python\Software\envs")
import tensorflow as tf
os.chdir("C:\Sreedhar\Python\Code\Deep Learning")
import numpy as np
from six.moves import cPickle as pickle
from six.moves import range
#-----------------------------------------------------------

w = tf.Variable(0.0)
b = tf.Variable(0.0)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
regr = w * x + b
init = tf.global_variables_initializer()

error = y - regr
loss = tf.reduce_sum(tf.square(error))

sess = tf.Session()
sess.run(init)
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
for i in range(1,5000):
    sess.run(train, {x:[1,2,3,4], y:[0, -1, -2, -3]})

print(sess.run([w, b]))
#-------------------------------------------------------------------------------------------
#https://www.tensorflow.org/get_started/mnist/beginners
#-------------------------------------------------------------------------------------------
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder(tf.float32, [None, 784]) 
W = tf.Variable(tf.zeros([784, 10]))
b = tf.zeros([10])

y = tf.matmul(x, W) + b
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y)
train_step= tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.arg_max(y,1), tf.arg_max(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))



