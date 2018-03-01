import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
import matplotlib.pyplot as plt
# read the data and labels as ont-hot vectors
# one-hot means a sparse vector for every observation where only
# the class label is 1, and every other class is 0.
# more info here:
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

n_input = 784
n_output = 10
n_hidden1 = 256
n_hidden2 = 128
n_hidden3 = 64
net_input = tf.placeholder(tf.float32, [None, n_input])
y_true = tf.placeholder(tf.float32, [None, 10])

W1 = tf.Variable(tf.truncated_normal ([n_input, n_hidden1]))
b1 = tf.Variable(tf.truncated_normal ([n_hidden1]))
W2 = tf.Variable(tf.truncated_normal ([n_hidden1, n_hidden2]))
b2 = tf.Variable(tf.truncated_normal ([n_hidden2]))
W3 = tf.Variable(tf.truncated_normal ([n_hidden2, n_hidden3]))
b3 = tf.Variable(tf.truncated_normal ([n_hidden3]))
W4 = tf.Variable(tf.truncated_normal ([n_hidden3, n_output]))
b4 = tf.Variable(tf.truncated_normal ([n_output]))

#W = tf.Variable(tf.truncated_normal ([n_input, n_output]))
#b = tf.Variable(tf.truncated_normal ([n_output]))


#the model
hidden1_res = tf.nn.tanh(tf.add(tf.matmul(net_input, W1), b1))
hidden2_res = tf.nn.tanh(tf.add(tf.matmul(hidden1_res, W2), b2))
hidden3_res = tf.nn.sigmoid(tf.add(tf.matmul(hidden2_res, W3), b3))
#hidden4_res = tf.nn.dropout(hidden3_res, 0.8)
#net_output = tf.nn.softmax(tf.add(tf.matmul(hidden3_res, W4), b4))
net_output = tf.nn.softmax(tf.add(tf.matmul(hidden3_res, W4), b4))
#net_output = tf.nn.softmax(tf.matmul(net_input, W) + b)

# prediction and actual using the argmax as the predicted label
correct_prediction = tf.equal(tf.argmax(net_output, 1), tf.argmax(y_true, 1))

# And now we can look at the mean of our network's correct guesses
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=net_output, labels=y_true))

eta = 0.02
#optimizer = tf.train.AdamOptimizer(eta).minimize(cost)
optimizer = tf.train.GradientDescentOptimizer(eta).minimize(cost)
#optimizer = tf.train.RMSPropOptimizer(eta).minimize(cost)