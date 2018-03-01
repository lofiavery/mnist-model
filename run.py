from ImageHandler import ImageHandler
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
import matplotlib.pyplot as plt

class Predictor(object):
    def predict(self,model_path = './model/the_model.ckpt'):
        n_input = 784
        n_output = 10
        n_hidden1 = 256
        n_hidden2 = 128
        n_hidden3 = 64
        net_input = tf.placeholder(tf.float32, [None, n_input])
        y_true = tf.placeholder(tf.float32, [None, 10])
        W1 = tf.Variable(tf.truncated_normal([n_input, n_hidden1]))
        b1 = tf.Variable(tf.truncated_normal([n_hidden1]))
        W2 = tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2]))
        b2 = tf.Variable(tf.truncated_normal([n_hidden2]))
        W3 = tf.Variable(tf.truncated_normal([n_hidden2, n_hidden3]))
        b3 = tf.Variable(tf.truncated_normal([n_hidden3]))
        W4 = tf.Variable(tf.truncated_normal([n_hidden3, n_output]))
        b4 = tf.Variable(tf.truncated_normal([n_output]))
        # the model
        hidden1_res = tf.nn.tanh(tf.add(tf.matmul(net_input, W1), b1))
        hidden2_res = tf.nn.tanh(tf.add(tf.matmul(hidden1_res, W2), b2))
        hidden3_res = tf.nn.sigmoid(tf.add(tf.matmul(hidden2_res, W3), b3))
        hidden3_res = tf.nn.dropout(hidden3_res, 0.8)
        net_output = tf.add(tf.matmul(hidden3_res, W4), b4)
        trained = tf.add(tf.matmul(hidden3_res, W4), b4)
        # prediction and actual using the argmax as the predicted label
        correct_prediction = tf.equal(tf.argmax(net_output, 1), tf.argmax(y_true, 1))
        # And now we can look at the mean of our network's correct guesses
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits = net_output, labels = y_true))
        eta = 0.02
        # optimizer = tf.train.AdamOptimizer(eta).minimize(cost)
        optimizer = tf.train.GradientDescentOptimizer(eta).minimize(cost)
        # optimizer = tf.train.RMSPropOptimizer(eta).minimize(cost)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        # ### load model from file ###
        with tf.Session() as sess:
            # Restore variables from disk.
             sess.run(tf.global_variables_initializer())
             saver = tf.train.import_meta_graph(model_path + '.meta')
             saver.restore(sess, model_path)
             parser = ImageHandler()
             images, labels = parser.parse_images()
             print("Accuracy for self uploaded images: {}".format(sess.run(accuracy,
                                                                          feed_dict = {
                                                                              net_input: images,
                                                                              y_true: labels
                                                                          })))

             pred = sess.run(trained, feed_dict = {net_input: images})
             return pred

if __name__ == "__main__":
    predictor = Predictor()
    predictions = predictor.predict()
    #print predictions
    for prediction in predictions:
        print("-----------------------------------------------------------")
        for p in prediction:
            print ("{:f}".format(float(p)))