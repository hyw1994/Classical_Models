'''This is the DenseNet model introduced by the paper "Deep Residual Learning for Image Recognition"'''
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from . import utils
from .model import model
from tensorflow.python.keras import layers
import re

# The network contains 50 layers within 5 groups.
# Initialization: 
# Weights: MSRA.
# Bias: 0.

class DenseNet121():
    def __init__(self, model_name, dataset_name, num_classes):
        super().__init__()

        self.num_classes = num_classes
        # Input image
        # self.x_image = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='x_image')
        # self.y_true_cls = tf.placeholder(tf.int64, shape=[None], name='y_true_cls')
        self.training_info = {'batch_size': 256, 'image_shape': 224, 'momentum': 0.9, 'l2lambda': 0.0001, 'learning_rate': 0.1, 'growth_rate': 12}
        # self.train_status = tf.placeholder(tf.bool)
        self.model_name = model_name
        self.writer = tf.summary.FileWriter('models/computational_graph/' + self.model_name + '/' + dataset_name)

    def build(self):        
        # Input layer
        input = layers.Input(shape=(224, 224, 3), name='Input')

        # Conv and max pool the input layer
        # input_conv = self.new_dense_layer(input_tensor=input, kernel_size=7, stride=2, name='Input', k=2)
        input_conv = layers.Conv2D(2*self.training_info.get('growth_rate'), kernel_size=7, strides=2, padding='SAME', kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(1e-4), use_bias=False, name='Input-conv')(input)
        input_bn = layers.BatchNormalization(beta_regularizer=tf.keras.regularizers.l2(1e-4), gamma_regularizer=tf.keras.regularizers.l2(1e-4), name='Input-bn')(input_conv)
        input_relu = layers.ReLU(name='Input-relu')(input_bn)
        input_pool = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='SAME', name='Input-max_pool')(input_relu)

        # Build dense block 1 and transition layer 1.
        dense_block1 = self.new_dense_block(input_tensor=input_pool, filters=6, name='dense_block1_')
        transition_layer1 = self.new_transition_layer(input_tensor=dense_block1, name='transition1_')

        # Build dense block 2 and transition layer 2.
        dense_block2 = self.new_dense_block(input_tensor=transition_layer1, filters=12, name='dense_block2_')
        transition_layer2 = self.new_transition_layer(input_tensor=dense_block2, name='transition2_')

        # Build dense block 3 and transition layer 3.
        dense_block3 = self.new_dense_block(input_tensor=transition_layer2, filters=24, name='dense_block3_')
        transition_layer3 = self.new_transition_layer(input_tensor=dense_block3, name='transition3_')

        # Build dense block 4 and transition layer 4.
        dense_block4 = self.new_dense_block(input_tensor=transition_layer3, filters=16, name='dense_block4_')
        dense_block4 = layers.BatchNormalization(beta_regularizer=tf.keras.regularizers.l2(1e-4), gamma_regularizer=tf.keras.regularizers.l2(1e-4), name='dense_block4_bn')(dense_block4)
        dense_block4 = layers.ReLU(name='dense_block4_relu')(dense_block4)

        # Classification layer.
        global_pool = layers.GlobalAveragePooling2D(name='global_average_pool')(dense_block4)
        dense_layer = layers.Dense(units=self.num_classes, activation='softmax')(global_pool)
        
        # Build the keras model.
        self.model = tf.keras.Model(inputs=input, outputs=dense_layer)
        self.model.compile(optimizer=tf.keras.optimizers.SGD(0.1, momentum=0.9, nesterov=True),
                                loss='sparse_categorical_crossentropy',
                                metrics=['sparse_categorical_accuracy'],
                                )
        self.model.summary()

    def train(self, EPOCH, iter_number, train_ds):
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1,
                                        patience=10, min_lr=0.0001)
        self.model.fit(train_ds, steps_per_epoch=iter_number ,epochs=EPOCH, callbacks=[reduce_lr])

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

    def new_dense_block(self, input_tensor, filters, name):
        '''This method will create dense block with 1*1 and 3*3 dense conv layers'''
        
        for i in range(filters):
            x = self.new_dense_layer(input_tensor=input_tensor, kernel_size=1, k=4, name=name+'conv-1-'+str(i+1))
            merge_tensor = self.new_dense_layer(input_tensor=x, kernel_size=3, name=name+'conv-3-'+str(i+1))
            input_tensor = layers.Concatenate()([input_tensor, merge_tensor])
        return input_tensor

    def new_dense_layer(self, input_tensor, kernel_size, name, stride=1, k=1):
        '''This method build the composit function with three operation: 
        BN, 
        ReLu,
        Conv2D.
        '''
        x = layers.BatchNormalization(beta_regularizer=tf.keras.regularizers.l2(1e-4), gamma_regularizer=tf.keras.regularizers.l2(1e-4), name=name+'-bn')(input_tensor)
        x = layers.ReLU(name=name+'-relu')(x)
        x = layers.Conv2D(filters=k*self.training_info.get('growth_rate'),kernel_initializer=tf.keras.initializers.he_uniform(), kernel_size=kernel_size, strides=stride, padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(1e-4), name=name+'-conv')(x)
        return x
    
    def new_transition_layer(self, input_tensor, name):
        '''This method build the transition layer to help the forward process with three operations: 
        BN,
        Conv2D,
        AveragePooling2D.
        '''
        filters = input_tensor.shape.as_list()[-1]
        x = layers.BatchNormalization(beta_regularizer=tf.keras.regularizers.l2(1e-4), gamma_regularizer=tf.keras.regularizers.l2(1e-4), name=name+'-bn')(input_tensor)
        x = layers.ReLU(name=name+'-relu')(x)
        x = layers.Conv2D(filters=int(0.5*filters), kernel_size=1, padding='SAME', kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(1e-4), name=name+'-conv')(x)
        x = layers.AveragePooling2D(pool_size=(2, 2),name=name+'-avg_pool')(x)
        return x