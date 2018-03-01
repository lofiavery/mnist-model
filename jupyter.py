from network import *
from ImageHandler import ImageHandler
import sys
SRC = None
DEST = None

class Learner(object):
    def __init__(self):
        pass
    def show_plot(self,l_loss):
        plt.title('Model Acuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.plot(l_loss, color = 'm')
        plt.show()
    def train(self, model_path):
        ## cut
        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        batch_size = 128
        n_epochs = 30
        l_loss = list()
        for epoch_i in range(n_epochs):
            for batch_i in range(0, mnist.train.num_examples, batch_size):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                sess.run(optimizer, feed_dict = {
                    net_input: batch_xs,
                    y_true: batch_ys
                })
            loss = sess.run(accuracy, feed_dict = {
                net_input: mnist.validation.images,
                y_true: mnist.validation.labels
            })
            print('Validation accuracy for epoch {} is: {}'.format(epoch_i + 1, loss))
            l_loss.append(loss)

        #save model
        save_path = saver.save(sess, model_path)
        #test predictions
        handler = ImageHandler()
        images,labels = handler.parse_images()
        f = {net_input: images}
        predictions = sess.run(net_output, feed_dict = f)
        self.print_predictions(predictions)
        print("Accuracy for test set: {}".format(sess.run(accuracy,
                                                          feed_dict = {
                                                              net_input: images,
                                                              y_true: labels
                                                          })))
    def print_predictions(self,predictions):
        for preds in predictions:
            print("--------------------------")
            for p in preds:
                print("{:f}".format(float(p)))
    def predict(self,load_path):
        new_saver = tf.train.Saver()
        #graph = tf.Graph()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            new_saver = tf.train.import_meta_graph(load_path+'.meta')
            new_saver.restore(sess, load_path)
            parser = ImageHandler()
            if(SRC != None and DEST != None):
                images, labels = parser.parse_images(SRC,DEST)
            else:
                images, labels = parser.parse_images()
            print("Accuracy for self uploaded images: {}".format(sess.run(accuracy,
                                                                          feed_dict = {
                                                                              net_input: images,
                                                                              y_true: labels
                                                                          })))
            predictions = sess.run(net_output, feed_dict = {net_input: images})
            #self.print_predictions(predictions)
        self.print_to_console(labels, predictions)
    def print_to_console(self,labels,predictions):
        for i in range(0,len(labels)):
            lab = self.get_max_index(labels[i])
            preds = self.get_max_index(predictions[i])
            print ("Number {0} classified as: {1}".format(lab['index'],preds['index']))

    def get_max_index(self,l):
        max = 0
        index = -1
        for i in range(0,len(l)):
            temp = l[i]
            if temp > max:
                max = temp
                index = i
        return {"max":max,"index":index}
# imgs_dir = "D:\\MLIternshipMihlala\\exe1\\test_mnist\\"
# processed = "D:\\MLIternshipMihlala\\exe1\\processed_mnist\\"
if __name__ == "__main__":
    model_path = "./models/model1.ckpt"
    #Learner().train(model_path)
    # input {images_src, images_dest}
    if len(sys.argv) == 3:
        SRC = sys.argv[1]
        DEST = sys.argv[2]
    Learner().predict(model_path)
