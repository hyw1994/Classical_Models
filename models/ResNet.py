'''This is the ResNet model introduced by the paper "Deep Residual Learning for Image Recognition"'''
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from . import utils
from .model import model
import re

# The network contains 50 layers within 5 groups.
# Initialization: 
# Weights: MSRA.
# Bias: 0.

class ResNet50():
    def __init__(self, model_name, dataset_name, num_classes):
        super().__init__()

        self.num_classes = num_classes
        # Input image
        self.x_image = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='x_image')
        self.y_true_cls = tf.placeholder(tf.int64, shape=[None], name='y_true_cls')
        self.training_info = {'batch_size': 256, 'image_shape': 224, 'momentum': 0.9, 'l2lambda': 0.0001, 'learning_rate': 0.1}
        self.ema = tf.train.ExponentialMovingAverage(decay=0.9999)
        self.train_status = tf.placeholder(tf.bool)
        self.best_validation_accuracy = 0
        self.model_name = model_name
        self.writer = tf.summary.FileWriter('models/computational_graph/' + self.model_name + '/' + dataset_name)

    def build(self):
        self.y_true = tf.one_hot(self.y_true_cls, depth=self.num_classes, axis=1, name='y_true')
        self.l2_loss = 0
        self.layers_collection = []
        self.weights_collection = []
        
        # First conv layer
        with tf.name_scope('conv_block1'):
            self.conv1, _ = utils.new_conv_layer(input=self.x_image,
                                            num_input_channels=3,
                                            filter_size=7,
                                            num_filters=64,
                                            stride=2,
                                            use_relu=False,
                                            use_MSRA=True,
                                            layers_collection=self.layers_collection,
                                            weights_collection=self.weights_collection,
                                            padding_mode='SAME',
                                            name='conv1')
            # self.conv1 = utils.batch_normalization(input_tensor=self.conv1, name='conv1-bn', train_status=self.train_status, ema=self.ema)
            self.conv1 = tf.layers.batch_normalization(inputs=self.conv1, training=self.train_status, name='conv1-bn')

            self.conv1 = tf.nn.relu(self.conv1, name='conv1-relu')
            self.conv1 = tf.nn.max_pool(self.conv1, (1, 3, 3, 1), (1, 2, 2, 1), padding='SAME', name='max_pool')
            self.layers_collection.append(self.conv1)

        with tf.name_scope('conv_block2'):
            self.conv2_1 = utils.new_conv_block(input_tensor=self.conv1,
                                              input_channel=64,
                                              kernel_size=3,
                                              filters=(64, 64, 256),
                                              stage='stage2',
                                              block='a',
                                              layers_collection=self.layers_collection,
                                              weights_collection=self.weights_collection,
                                              ema=self.ema,
                                              train_status=self.train_status,
                                              stride=1)

            self.conv2_2 = utils.new_identity_block(input_tensor=self.conv2_1,
                                              input_channel=256,
                                              kernel_size=3,
                                              filters=(64, 64, 256),
                                              stage='stage2',
                                              block='b',
                                              layers_collection=self.layers_collection,
                                              weights_collection=self.weights_collection,
                                              ema=self.ema,
                                              train_status=self.train_status,
                                              )

            self.conv2_3 = utils.new_identity_block(input_tensor=self.conv2_2,
                                              input_channel=256,
                                              kernel_size=3,
                                              filters=(64, 64, 256),
                                              stage='stage2',
                                              block='c',
                                              layers_collection=self.layers_collection,
                                              weights_collection=self.weights_collection,
                                              ema=self.ema,
                                              train_status=self.train_status,
                                              )

        with tf.name_scope('conv_block3'):
            self.conv3_1 = utils.new_conv_block(input_tensor=self.conv2_3,
                                              input_channel=256,
                                              kernel_size=3,
                                              filters=(128, 128, 512),
                                              stage='stage3',
                                              block='a',
                                              layers_collection=self.layers_collection,
                                              weights_collection=self.weights_collection,
                                              ema=self.ema,
                                              train_status=self.train_status
                                              )

            self.conv3_2 = utils.new_identity_block(input_tensor=self.conv3_1,
                                              input_channel=512,
                                              kernel_size=3,
                                              filters=(128, 128, 512),
                                              stage='stage3',
                                              block='b',
                                              layers_collection=self.layers_collection,
                                              weights_collection=self.weights_collection,
                                              ema=self.ema,
                                              train_status=self.train_status,
                                              )

            self.conv3_3 = utils.new_identity_block(input_tensor=self.conv3_2,
                                              input_channel=512,
                                              kernel_size=3,
                                              filters=(128, 128, 512),
                                              stage='stage3',
                                              block='c',
                                              layers_collection=self.layers_collection,
                                              weights_collection=self.weights_collection,
                                              ema=self.ema,
                                              train_status=self.train_status,
                                              )

            self.conv3_4 = utils.new_identity_block(input_tensor=self.conv3_3,
                                              input_channel=512,
                                              kernel_size=3,
                                              filters=(128, 128, 512),
                                              stage='stage3',
                                              block='d',
                                              layers_collection=self.layers_collection,
                                              weights_collection=self.weights_collection,
                                              ema=self.ema,
                                              train_status=self.train_status,
                                              )

        with tf.name_scope('conv_block4'):
            self.conv4_1 = utils.new_conv_block(input_tensor=self.conv3_4,
                                            input_channel=512,
                                            kernel_size=3,
                                            filters=(256, 256, 1024),
                                            stage='stage4',
                                            block='a',
                                            layers_collection=self.layers_collection,
                                            weights_collection=self.weights_collection,
                                            ema=self.ema,
                                            train_status=self.train_status
                                            )

            self.conv4_2 = utils.new_identity_block(input_tensor=self.conv4_1,
                                            input_channel=1024,
                                            kernel_size=3,
                                            filters=(256, 256, 1024),
                                            stage='stage4',
                                            block='b',
                                            layers_collection=self.layers_collection,
                                            weights_collection=self.weights_collection,
                                            ema=self.ema,
                                            train_status=self.train_status,
                                            )

            self.conv4_3 = utils.new_identity_block(input_tensor=self.conv4_2,
                                            input_channel=1024,
                                            kernel_size=3,
                                            filters=(256, 256, 1024),
                                            stage='stage4',
                                            block='c',
                                            layers_collection=self.layers_collection,
                                            weights_collection=self.weights_collection,
                                            ema=self.ema,
                                            train_status=self.train_status,
                                            )

            self.conv4_4 = utils.new_identity_block(input_tensor=self.conv4_3,
                                            input_channel=1024,
                                            kernel_size=3,
                                            filters=(256, 256, 1024),
                                            stage='stage4',
                                            block='d',
                                            layers_collection=self.layers_collection,
                                            weights_collection=self.weights_collection,
                                            ema=self.ema,
                                            train_status=self.train_status,
                                            )
            
            self.conv4_5 = utils.new_identity_block(input_tensor=self.conv4_4,
                                            input_channel=1024,
                                            kernel_size=3,
                                            filters=(256, 256, 1024),
                                            stage='stage4',
                                            block='e',
                                            layers_collection=self.layers_collection,
                                            weights_collection=self.weights_collection,
                                            ema=self.ema,
                                            train_status=self.train_status,
                                            )

            self.conv4_6 = utils.new_identity_block(input_tensor=self.conv4_5,
                                            input_channel=1024,
                                            kernel_size=3,
                                            filters=(256, 256, 1024),
                                            stage='stage4',
                                            block='f',
                                            layers_collection=self.layers_collection,
                                            weights_collection=self.weights_collection,
                                            ema=self.ema,
                                            train_status=self.train_status,
                                            )

        with tf.name_scope('conv_block5'):
            self.conv5_1 = utils.new_conv_block(input_tensor=self.conv4_6,
                                              input_channel=1024,
                                              kernel_size=3,
                                              filters=(512, 512, 2048),
                                              stage='stage5',
                                              block='a',
                                              layers_collection=self.layers_collection,
                                              weights_collection=self.weights_collection,
                                              ema=self.ema,
                                              train_status=self.train_status,
                                              )

            self.conv5_2 = utils.new_identity_block(input_tensor=self.conv5_1,
                                              input_channel=2048,
                                              kernel_size=3,
                                              filters=(512, 512, 2048),
                                              stage='stage5',
                                              block='b',
                                              layers_collection=self.layers_collection,
                                              weights_collection=self.weights_collection,
                                              ema=self.ema,
                                              train_status=self.train_status,
                                              )

            self.conv5_3 = utils.new_identity_block(input_tensor=self.conv5_2,
                                              input_channel=2048,
                                              kernel_size=3,
                                              filters=(512, 512, 2048),
                                              stage='stage5',
                                              block='c',
                                              layers_collection=self.layers_collection,
                                              weights_collection=self.weights_collection,
                                              ema=self.ema,
                                              train_status=self.train_status,
                                              )

        self.avg_pool = tf.nn.avg_pool(self.conv5_3, (1, 7, 7, 1), (1, 1, 1, 1), name='average_pool', padding='VALID')
        self.layers_collection.append(self.avg_pool)
        
        # Flatten the last CONV layer
        self.flatten, self.num_features = utils.flatten_layer(input=self.avg_pool, layers_collection=self.layers_collection)

        # Only one fully connected layer
        self.fc1_layer, _ = utils.new_fc_layer(input=self.flatten, num_inputs=self.num_features, num_outputs=self.num_classes, use_dropout=False, use_relu=False, use_xavier=True)

        with tf.name_scope('loss'):
            # Build loss function
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_true, logits=self.fc1_layer)
            # Calculate the l2 loss.
            for weight in self.weights_collection:
                self.l2_loss = self.l2_loss + tf.nn.l2_loss(weight)
            self.cost = tf.reduce_mean(self.cross_entropy) + self.training_info.get('l2lambda') * self.l2_loss

            tf.summary.scalar('cross_entropy_loss', self.cost)
        
        with tf.name_scope('optimizer'):
            # Build the optimizer
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.training_info.get('learning_rate'), momentum=self.training_info.get('momentum'), name='optimizer').minimize(loss=self.cost)

        with tf.name_scope('accuracy'):
            # Build the calculation to accuracy
            self.y_pred = tf.nn.softmax(self.fc1_layer)
            self.y_pred_cls = tf.argmax(self.y_pred, axis=1)
            self.correct_prediction = tf.equal(self.y_true_cls, self.y_pred_cls)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)
        
        print('-'*32)
        for layer in self.layers_collection:
            print(layer)
        print('-'*32)

        # Find all variables that is needed to save.
        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_variable = [g for g in g_list if "moving_mean" in g.name]
        bn_moving_variable += [g for g in g_list if "moving_variance" in g.name]
        var_list += bn_moving_variable
        self.saver = tf.train.Saver(var_list, max_to_keep=5)
        self.merged_summary = tf.summary.merge_all()
        self.train_op = tf.group(self.optimizer, tf.get_collection(tf.GraphKeys.UPDATE_OPS))

    def train_res(self, sess, EPOCH, iter_number, train_numpy, evaluation_numpy=None):
        sess.run(tf.global_variables_initializer())
        try:
            self.recover_params(sess)
            print("Lastest saved model loaded successfully!")
        except ValueError as identifier:
            print(identifier)
            print("The model will be trained from begining!")

        for epoch in range(EPOCH):
            print("-"*32)
            for step in range(iter_number):
                sess.graph.finalize()
                image_batch, label_batch = next(train_numpy)
                feed_dict_train = {self.x_image: image_batch, self.y_true_cls: label_batch, self.train_status: 1}
                feed_dict_test = {self.x_image: image_batch, self.y_true_cls: label_batch, self.train_status: 0}
                if step % 5 == 0:
                    s = sess.run(self.merged_summary, feed_dict=feed_dict_train)
                    self.writer.add_summary(s, step)
                _ = sess.run(self.train_op, feed_dict=feed_dict_train)
                train_accuracy, train_cost = sess.run([self.accuracy, self.cost], feed_dict=feed_dict_train)
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
        feed_dict = {self.x_image: image_batch, self.y_true_cls: label_batch, self.train_status: 0}
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