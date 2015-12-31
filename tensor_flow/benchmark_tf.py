###################################
#
# Benchmark Tensor Flow
# J. Gartner
#
# Train gradient descent and
# convoluted nueral network
# models on the MINST handwritten
# number dataset
#
# THIS CODE IS TAKEN VERBATUM
# FROM THE TENSOR FLOW TUTORIAL:
# https://www.tensorflow.org/versions/master/tutorials/mnist/pros/index.html
# I TAKE NO CREDIT AS THE CODE'S AUTHOR.
#
###################################

import tensorflow as tf
import input_data
from datetime import datetime

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def main():
    t0 = datetime.now()
    print "\n\n***************************************\n*\n* Initializing data\n*\n***************************************\n"
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    sess = tf.InteractiveSession()
    x = tf.placeholder("float", shape=[None, 784])
    y_ = tf.placeholder("float", shape=[None, 10])
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))
    sess.run(tf.initialize_all_variables())
    y = tf.nn.softmax(tf.matmul(x,W) + b)
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    t1 = datetime.now()
    print "\n\n***************************************\n*\n* Time to initialize data:", (t1-t0).total_seconds()
    print "* Training gradient descent model (gdm)\n*\n***************************************"
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    for i in range(1000):
        batch = mnist.train.next_batch(50)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    t2 = datetime.now()
    print "\n\n***************************************\n*\n* Time to train gdm:", (t2-t1).total_seconds()
    print "* Evaluating gdm\n*\n***************************************"
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    t3 = datetime.now()
    print "\n\n***************************************\n*\n* Time to evaluate gdm:", (t3-t2).total_seconds()
    print "* Preparing convoluted neural network (cnn)\n*\n***************************************"
    # Init vars
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    # First convolution Layer
    x_image = tf.reshape(x, [-1,28,28,1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    # Second convolution Layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    # Densely Connected Layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    # Readout Layer
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    t4 = datetime.now()
    print "\n\n***************************************\n*\n* Time to prepare cnn:", (t4-t3).total_seconds()
    print "* Train cnn\n*\n***************************************"
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    sess.run(tf.initialize_all_variables())
    for i in range(2000):
      batch = mnist.train.next_batch(50)
      if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    t5 = datetime.now()
    print "\n\n***************************************\n*\n* Time to train cnn:", (t5-t4).total_seconds()
    print "* Apply cnn\n*\n***************************************"
    print("test accuracy %g"%accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    t6 = datetime.now()
    print "\n\n***************************************\n*\n* Time to apply cnn:", (t6-t5).total_seconds()
    print "*\n***************************************"


if __name__ == "__main__":
    main()