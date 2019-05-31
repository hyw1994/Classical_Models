'''This is the model class that defines the general methods to train and load model.'''
import tensorflow as tf
import re
from . import utils

class model():
    def __init__(self, model_name, dataset_name):
        super().__init__()
        # Best validation accuracy.
        self.best_validation_accuracy = 0
        self.model_name = model_name
        self.writer = tf.summary.FileWriter('models/computational_graph/' + self.model_name + '/' + dataset_name)

    def build():
        return
    
    def train(self, sess, EPOCH, iter_number, train_numpy, evaluation_numpy=None):
        self.saver = tf.train.Saver(max_to_keep=5)
        sess.run(tf.global_variables_initializer())
        merged_summary = tf.summary.merge_all()

        try:
            self.recover_params(sess)
            print("Lastest saved model loaded successfully!")
        except ValueError as identifier:
            print(identifier)
            print("The model will be trained from begining!")

        for epoch in range(EPOCH):
            print("-"*32)
            for step in range(iter_number):
                image_batch, label_batch = next(train_numpy)
                feed_dict_train = {self.x_image: image_batch, self.y_true_cls: label_batch}
                feed_dict_test = {self.x_image: image_batch, self.y_true_cls: label_batch, self.dropout_rate: 0.0}
                if step % 5 == 0:
                    s = sess.run(merged_summary, feed_dict=feed_dict_train)
                    self.writer.add_summary(s, step)
                _ = sess.run(self.optimizer, feed_dict=feed_dict_train)
                train_accuracy, train_cost = sess.run([self.accuracy, self.cost], feed_dict=feed_dict_test)
                print("EPOCH: {}, step: {}, train_batch_accuracy: {}, train_batch_loss: {}".format(epoch+1, step+1, train_accuracy, train_cost))
                if (step % 100 == 0) or (epoch+1 == EPOCH) and (step+1 == iter_number) :
                    if(evaluation_numpy is None):
                        evaluation_numpy = (image_batch, label_batch)
                    evaluation_accuracy = self.save_params(sess, global_step=epoch * iter_number + step+1, evaluation_numpy=evaluation_numpy)
                    print("evaluation_accuracy: {}".format(evaluation_accuracy))
                print('-'*32)
            print("-"*32)
    
    def save_graph(self, sess):
        '''Save the computational graph to tensorboard'''
        self.writer.add_graph(sess.graph)

    def evaluate_accuracy(self, sess, data_numpy):
        '''Evalute the accuracy for the given dataset(especially for cross-validation set and test set).'''
        image_batch, label_batch = data_numpy
        feed_dict = {self.x_image: image_batch, self.y_true_cls: label_batch, self.dropout_rate: 0.0}
        accuracy = sess.run(self.accuracy, feed_dict=feed_dict)
        return accuracy    

    def save_params(self, sess, global_step, evaluation_numpy):
        '''Save trained tf.Variable parameters to file'''
        evaluation_accuracy = self.evaluate_accuracy(sess, evaluation_numpy)
        if(evaluation_accuracy > self.best_validation_accuracy):
            utils.save_params(sess=sess, file_name=self.model_name + '/model', global_step=global_step, saver=self.saver)
            self.best_validation_accuracy = evaluation_accuracy
        return evaluation_accuracy

    def recover_params(self, sess):
        '''Recover the model params from latest save point.'''
        prog = re.compile(r'model-[0-9]*')
        utils.load_params(sess=sess, model_name=self.model_name, prog=prog, saver=self.saver)