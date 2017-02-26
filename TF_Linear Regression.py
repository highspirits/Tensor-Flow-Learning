# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 19:50:33 2017

@author: vsnalam
"""

#Example from Getting Started with Tensorflow book by Giancario Zaccone, Packt Publishing

import os
os.chdir("C:\Sreedhar\Software\Anaconda Python\Software\envs")
import tensorflow as tf

os.chdir("C:\Sreedhar\Python\Code\Deep Learning\TensorFlow Learning")

#--------------------------------------------------------------------------------------
#Linear Regression using TensorFlow
#--------------------------------------------------------------------------------------

import numpy as np

#define the number of points we want to draw:
number_of_points = 500

#initialize the dependent and independent variables
x_point = []
y_point = []

#set arbitary constants
a = 0.45
b = 0.87

#generate 300 random points around the regression equation y = 0.22x + 0.78, with some noise

for i in range(number_of_points):
    x = np.random.normal(0.0, 0.5)
    y = a * x + b + np.random.normal(0.0, 0.1)
    x_point.append([x])
    y_point.append([y])
    
#View the points generated
import matplotlib.pyplot as plt
plt.plot(x_point, y_point, 'o', label = 'Input Data')
plt.legend()
plt.show()

#Cost function and gradient descent
#We define a linear relation between x_point and y_point as - y = A * x_point + b. 
#We now use gradient descent and tensor flow to identify the values of A and b.
#Define an arbitary value for A
A = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
#Initialize b to zero
b = tf.Variable(tf.zeros([1]))
y = A * x_point + b
cost_function = tf.reduce_mean(tf.square(y - y_point))
optimizer = tf.train.GradientDescentOptimizer(0.5)
#train = tf.train.GradientDescentOptimizer(0.5).minimize(cost_function)
train = optimizer.minimize(cost_function)
#--------------------
model = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(model)
    for step in range(0, 21):
        session.run(train)
        if (step % 5 == 0):
            plt.plot(x_point, y_point, 'o', label = 'step = {}'.format(step))
            plt.plot(x_point, session.run(A)*x_point+session.run(b))
            plt.legend()
            plt.show()
            print(session.run(A))
            print(session.run(b))
            
    
    












