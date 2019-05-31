''' This is the VGG model from the original paper for study purpose.'''
import tensorflow as tf
import csv
import matplotlib.pyplot as plt
import numpy as np
from . import utils
from .model import model

# VGG network architecture.
# Initialization: 
# Weights: Xavier, 0 mean, 0.01 variation, .
# Bias: 0.

# [FILTER SHAPE], [OUTPUT SHAPE]
# []              [224, 224, 3]   INPUT LAYER: Subtracting the mean RGB value for training set. Remember the mean values for evaluation.
# [64, 3, 3]      [224, 224, 64]  CONV1_1 LAYER: 64 filters with size [3, 3] and stride=1, pad=1, activation=ReLU.
# [64, 3, 3]      [224, 224, 64]  CONV1_2 LAYER: 64 filters with size [3, 3] and stride=1, pad=1, activation=ReLU.

# [2, 2]          [112, 112, 64]  MAX POOLING1 LAYER: size=2, stride=2.

# [128, 3, 3]     [112, 112, 128]  CONV2_1 LAYER: 128 filters with size [3, 3] and stride=1, pad=1, activation=ReLU.
# [128, 3, 3]     [112, 112, 128]  CONV2_2 LAYER: 128 filters with size [3, 3] and stride=1, pad=1, activation=ReLU.

# [2, 2]          [56, 56, 128]    MAX POOLING2 LAYER: size=2, stride=2.

# [256, 3, 3]     [56, 56, 256]   CONV3_1 LAYER: 256 filters with size [3, 3] and stride=1, pad=1, activation=ReLU.
# [256, 3, 3]     [56, 56, 256]   CONV3_2 LAYER: 256 filters with size [3, 3] and stride=1, pad=1, activation=ReLU.
# [256, 3, 3]     [56, 56, 256]   CONV3_3 LAYER: 256 filters with size [3, 3] and stride=1, pad=1, activation=ReLU.
# !(VGG19)
# [256, 3, 3]     [56, 56, 256]   CONV3_4 LAYER: 256 filters with size [3, 3] and stride=1, pad=1, activation=ReLU.

# [2, 2]          [28, 28, 256]   MAX POOLING3 LAYER: size=2, stride=2.

# [512, 3, 3]     [28, 28, 512]   CONV4_1 LAYER: 512 filters with size [3, 3] and stride=1, pad=1, activation=ReLU.
# [512, 3, 3]     [28, 28, 512]   CONV4_2 LAYER: 512 filters with size [3, 3] and stride=1, pad=1, activation=ReLU.
# [512, 3, 3]     [28, 28, 512]   CONV4_3 LAYER: 512 filters with size [3, 3] and stride=1, pad=1, activation=ReLU.
# !(VGG19)
# [512, 3, 3]     [28, 28, 512]   CONV4_4 LAYER: 512 filters with size [3, 3] and stride=1, pad=1, activation=ReLU.

# [2, 2]          [14, 14, 512]   MAX POOLING4 LAYER: size=2, stride=2.

# [512, 3, 3]     [14, 14, 512]   CONV5_1 LAYER: 512 filters with size [3, 3] and stride=1, pad=1, activation=ReLU.
# [512, 3, 3]     [14, 14, 512]   CONV5_2 LAYER: 512 filters with size [3, 3] and stride=1, pad=1, activation=ReLU.
# [512, 3, 3]     [14, 14, 512]   CONV5_3 LAYER: 512 filters with size [3, 3] and stride=1, pad=1, activation=ReLU.
# !(VGG19)
# [512, 3, 3]     [14, 14, 512]   CONV5_4 LAYER: 512 filters with size [3, 3] and stride=1, pad=1, activation=ReLU.

# [2, 2]          [7, 7, 512]     MAX POOLING5 LAYER: size=2, stride=2.

# [25088]                         FLATTEN LAYER: 7*7*512

# [4096]          [4096]          FC1: 4096 neurons, ReLU, dropout=0.5.
# [4096]          [4096]          FC2: 4096 neurons, ReLU, dropout=0.5.
# [1000]          [1000]          FC3: 1000 neurons, Softmax. change it to the categories size while using.

class VGG(model):
    def __init__(self, dataset_name, num_classes, model_type='VGG16'):
        super().__init__(model_name=model_type, dataset_name=dataset_name)
        # Specify the network type, VGG16 or VGG19.
        '''Build the network with the type given (VGG16 or VGG19).'''
        if(model_type == 'VGG16' or model_type == 'vgg16' or model_type == 'VGG19' or model_type == 'vgg19'):
            self.model_type = model_type
        else:
            raise ValueError("Unsupported model type: {}, use 'VGG16' or 'VGG19' only.".format(type))

        self.num_classes = num_classes
        # Input image.
        self.x_image = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='x_image')
        self.y_true_cls = tf.placeholder(tf.int64, shape=[None], name='y_true_cls')
        
        self.training_info = {'batch_size': 256, 'image_shape': 224, 'momentum': 0.9, 'l2lambda': 5e-4, 'learning_rate': 1e-2}
        self.layer_info = {'paddings': 1, 'conv_size':3, 'conv_stride': 1, 'pool_size':2, 'pool_stride': 2, 'bias_value': 0, 'use_xavier': True}
        # Conv1 layer
        self.conv1 = {'num_filters': 64, 'vgg16_num': 2, 'vgg19_num': 2}
        # Conv2 layer
        self.conv2 = {'num_filters': 128, 'vgg16_num': 2, 'vgg19_num': 2}
        # Conv3 layer
        self.conv3 = {'num_filters': 256, 'vgg16_num': 3, 'vgg19_num': 4}
        # Conv4 layer
        self.conv4 = {'num_filters': 512, 'vgg16_num': 3, 'vgg19_num': 4}
        # Conv5 layer
        self.conv5 = {'num_filters': 512, 'vgg16_num': 3, 'vgg19_num': 4}
        # FC1 layer
        self.fc1 = 4096
        self.fc2 = 4096
        self.fc3 = self.num_classes

    def build(self):
        print("The model in use is: {}".format(self.model_type))
        self.y_true = tf.one_hot(self.y_true_cls, depth=self.num_classes, axis=1, name='y_true')
        self.l2_loss = 0
        self.layers_collection = []
        self.weights_collection = []

        # Build conv1 layer.
        self.conv1_1_layer, self.weight1_1 = utils.new_conv_layer(input=self.x_image,
                                                  num_input_channels=3,
                                                  paddings=self.layer_info.get('paddings'),
                                                  filter_size=self.layer_info.get('conv_size'),
                                                  stride=self.layer_info.get('conv_stride'),
                                                  num_filters=self.conv1.get('num_filters'),
                                                  use_xavier=True,
                                                  layers_collection = self.layers_collection,
                                                  weights_collection = self.weights_collection,
                                                  name='conv1_1')

        self.conv1_2_layer, self.weight1_2 = utils.new_conv_layer(input=self.conv1_1_layer,
                                                  num_input_channels=self.conv1.get('num_filters'),
                                                  paddings=self.layer_info.get('paddings'),
                                                  filter_size=self.layer_info.get('conv_size'),
                                                  stride=self.layer_info.get('conv_stride'),
                                                  num_filters=self.conv1.get('num_filters'),
                                                  use_xavier=True,
                                                  layers_collection = self.layers_collection,
                                                  weights_collection = self.weights_collection,
                                                  name='conv1_2')
        
        self.pool1_layer = utils.new_pooling_layer(input=self.conv1_2_layer, 
                                                   pool_size=self.layer_info.get('pool_size'),
                                                   pool_stride=self.layer_info.get('pool_stride'),
                                                   layers_collection = self.layers_collection,
                                                   name='pool1')

        # Build conv2 layer.
        self.conv2_1_layer, self.weight2_1 = utils.new_conv_layer(input=self.pool1_layer,
                                                  num_input_channels=self.conv1.get('num_filters'),
                                                  paddings=self.layer_info.get('paddings'),
                                                  filter_size=self.layer_info.get('conv_size'),
                                                  stride=self.layer_info.get('conv_stride'),
                                                  num_filters=self.conv2.get('num_filters'),
                                                  use_xavier=True,
                                                  layers_collection = self.layers_collection,
                                                  weights_collection = self.weights_collection,
                                                  name='conv2_1')
        
        self.conv2_2_layer, self.weight2_2 = utils.new_conv_layer(input=self.conv2_1_layer,
                                                  num_input_channels=self.conv2.get('num_filters'),
                                                  paddings=self.layer_info.get('paddings'),
                                                  filter_size=self.layer_info.get('conv_size'),
                                                  stride=self.layer_info.get('conv_stride'),
                                                  num_filters=self.conv2.get('num_filters'),
                                                  use_xavier=True,
                                                  layers_collection = self.layers_collection,
                                                  weights_collection = self.weights_collection,
                                                  name='conv2_2')

        self.pool2_layer = utils.new_pooling_layer(input=self.conv2_2_layer, 
                                                   pool_size=self.layer_info.get('pool_size'),
                                                   pool_stride=self.layer_info.get('pool_stride'),
                                                   layers_collection = self.layers_collection,
                                                   name='pool2')

        # Build conv3 layer.
        self.conv3_1_layer, self.weight3_1 = utils.new_conv_layer(input=self.pool2_layer,
                                                                  num_input_channels=self.conv2.get('num_filters'),
                                                                  paddings=self.layer_info.get('paddings'),
                                                                  filter_size=self.layer_info.get('conv_size'),
                                                                  stride=self.layer_info.get('conv_stride'),
                                                                  num_filters=self.conv3.get('num_filters'),
                                                                  use_xavier=True,
                                                                  layers_collection = self.layers_collection,
                                                                  weights_collection = self.weights_collection,
                                                                  name='conv3_1')
        
        self.conv3_2_layer, self.weight3_2 = utils.new_conv_layer(input=self.conv3_1_layer,
                                                  num_input_channels=self.conv3.get('num_filters'),
                                                  paddings=self.layer_info.get('paddings'),
                                                  filter_size=self.layer_info.get('conv_size'),
                                                  stride=self.layer_info.get('conv_stride'),
                                                  num_filters=self.conv3.get('num_filters'),
                                                  use_xavier=True,
                                                  layers_collection = self.layers_collection,
                                                  weights_collection = self.weights_collection,
                                                  name='conv3_2')
                                                  
        self.conv3_3_layer, self.weight3_3 = utils.new_conv_layer(input=self.conv3_2_layer,
                                                  num_input_channels=self.conv3.get('num_filters'),
                                                  paddings=self.layer_info.get('paddings'),
                                                  filter_size=self.layer_info.get('conv_size'),
                                                  stride=self.layer_info.get('conv_stride'),
                                                  num_filters=self.conv3.get('num_filters'),
                                                  use_xavier=True,
                                                  layers_collection = self.layers_collection,
                                                  weights_collection = self.weights_collection,
                                                  name='conv3_3')
        self.conv3_output = self.conv3_3_layer
        
        if(self.model_type == 'VGG19' or self.model_type == 'vgg19'):
            self.conv3_4_layer, self.weight3_4 = utils.new_conv_layer(input=self.conv3_3_layer,
                                                      num_input_channels=self.conv3.get('num_filters'),
                                                      paddings=self.layer_info.get('paddings'),
                                                      filter_size=self.layer_info.get('conv_size'),
                                                      stride=self.layer_info.get('conv_stride'),
                                                      num_filters=self.conv3.get('num_filters'),
                                                      use_xavier=True,
                                                      layers_collection = self.layers_collection,
                                                      weights_collection = self.weights_collection,
                                                      name='conv3_4')
            self.conv3_output = self.conv3_4_layer

        self.pool3_layer = utils.new_pooling_layer(input=self.conv3_output, 
                                                   pool_size=self.layer_info.get('pool_size'),
                                                   pool_stride=self.layer_info.get('pool_stride'),
                                                   layers_collection = self.layers_collection,
                                                   name='pool3')

        # Build conv4 layer.
        self.conv4_1_layer, self.weight4_1 = utils.new_conv_layer(input=self.pool3_layer,
                                                  num_input_channels=self.conv3.get('num_filters'),
                                                  paddings=self.layer_info.get('paddings'),
                                                  filter_size=self.layer_info.get('conv_size'),
                                                  stride=self.layer_info.get('conv_stride'),
                                                  num_filters=self.conv4.get('num_filters'),
                                                  use_xavier=True,
                                                  layers_collection = self.layers_collection,
                                                  weights_collection = self.weights_collection,
                                                  name='conv4_1')
        
        self.conv4_2_layer, self.weight4_2 = utils.new_conv_layer(input=self.conv4_1_layer,
                                                  num_input_channels=self.conv4.get('num_filters'),
                                                  paddings=self.layer_info.get('paddings'),
                                                  filter_size=self.layer_info.get('conv_size'),
                                                  stride=self.layer_info.get('conv_stride'),
                                                  num_filters=self.conv4.get('num_filters'),
                                                  use_xavier=True,
                                                  layers_collection = self.layers_collection,
                                                  weights_collection = self.weights_collection,
                                                  name='conv4_2')
                                                  
        self.conv4_3_layer, self.weight4_3 = utils.new_conv_layer(input=self.conv4_2_layer,
                                                  num_input_channels=self.conv4.get('num_filters'),
                                                  paddings=self.layer_info.get('paddings'),
                                                  filter_size=self.layer_info.get('conv_size'),
                                                  stride=self.layer_info.get('conv_stride'),
                                                  num_filters=self.conv4.get('num_filters'),
                                                  use_xavier=True,
                                                  layers_collection = self.layers_collection,
                                                  weights_collection = self.weights_collection,
                                                  name='conv4_3')
        self.conv4_output = self.conv4_3_layer

        if(self.model_type == 'VGG19' or self.model_type == 'vgg19'):
            self.conv4_4_layer, self.weight4_4 = utils.new_conv_layer(input=self.conv4_3_layer,
                                                                      num_input_channels=self.conv4.get('num_filters'),
                                                                      paddings=self.layer_info.get('paddings'),
                                                                      filter_size=self.layer_info.get('conv_size'),
                                                                      stride=self.layer_info.get('conv_stride'),
                                                                      num_filters=self.conv4.get('num_filters'),
                                                                      use_xavier=True,
                                                                      layers_collection = self.layers_collection,
                                                                      weights_collection = self.weights_collection,
                                                                      name='conv4_4')
            self.conv4_output = self.conv4_4_layer

        self.pool4_layer = utils.new_pooling_layer(input=self.conv4_output, 
                                                   pool_size=self.layer_info.get('pool_size'),
                                                   pool_stride=self.layer_info.get('pool_stride'),
                                                   layers_collection = self.layers_collection,
                                                   name='pool4')
        
        # Build conv5 layer.
        self.conv5_1_layer, self.weight5_1 = utils.new_conv_layer(input=self.pool4_layer,
                                                                  num_input_channels=self.conv4.get('num_filters'),
                                                                  paddings=self.layer_info.get('paddings'),
                                                                  filter_size=self.layer_info.get('conv_size'),
                                                                  stride=self.layer_info.get('conv_stride'),
                                                                  num_filters=self.conv5.get('num_filters'),
                                                                  use_xavier=True,
                                                                  layers_collection = self.layers_collection,
                                                                  weights_collection = self.weights_collection,
                                                                  name='conv5_1')
        
        self.conv5_2_layer, self.weight5_2 = utils.new_conv_layer(input=self.conv5_1_layer,
                                                                  num_input_channels=self.conv5.get('num_filters'),
                                                                  paddings=self.layer_info.get('paddings'),
                                                                  filter_size=self.layer_info.get('conv_size'),
                                                                  stride=self.layer_info.get('conv_stride'),
                                                                  num_filters=self.conv5.get('num_filters'),
                                                                  use_xavier=True,
                                                                  layers_collection = self.layers_collection,
                                                                  weights_collection = self.weights_collection,
                                                                  name='conv5_2')
                                                  
        self.conv5_3_layer, self.weight5_3 = utils.new_conv_layer(input=self.conv5_2_layer,
                                                                  num_input_channels=self.conv5.get('num_filters'),
                                                                  paddings=self.layer_info.get('paddings'),
                                                                  filter_size=self.layer_info.get('conv_size'),
                                                                  stride=self.layer_info.get('conv_stride'),
                                                                  num_filters=self.conv5.get('num_filters'),
                                                                  use_xavier=True,
                                                                  layers_collection = self.layers_collection,
                                                                  weights_collection = self.weights_collection,
                                                                  name='conv5_3')
        self.conv5_output = self.conv5_3_layer

        if(self.model_type == 'VGG19' or self.model_type == 'vgg19'):
            self.conv5_4_layer, self.weight5_4 = utils.new_conv_layer(input=self.conv5_3_layer,
                                                                      num_input_channels=self.conv5.get('num_filters'),
                                                                      paddings=self.layer_info.get('paddings'),
                                                                      filter_size=self.layer_info.get('conv_size'),
                                                                      stride=self.layer_info.get('conv_stride'),
                                                                      num_filters=self.conv5.get('num_filters'),
                                                                      use_xavier=True,
                                                                      layers_collection = self.layers_collection,
                                                                      weights_collection = self.weights_collection,
                                                                      name='conv5_4')
            self.conv5_output = self.conv5_4_layer

        self.pool5_layer = utils.new_pooling_layer(input=self.conv5_output, 
                                                   pool_size=self.layer_info.get('pool_size'),
                                                   pool_stride=self.layer_info.get('pool_stride'),
                                                   layers_collection = self.layers_collection,
                                                   name='pool5')
        
        # Build FC layers
        self.flatten_layer, self.num_features = utils.flatten_layer(input=self.pool5_layer, name='flatten', layers_collection = self.layers_collection)
        self.dropout_rate = tf.placeholder_with_default(0.5, shape=())
        self.fc1_layer, self.weight_fc1 = utils.new_fc_layer(input=self.flatten_layer, num_inputs=self.num_features, num_outputs=self.fc1, dropout_rate=self.dropout_rate,use_xavier=True, layers_collection=self.layers_collection, weights_collection=self.weights_collection, name='fc1')
        self.fc2_layer, self.weight_fc2 = utils.new_fc_layer(input=self.fc1_layer, num_inputs=self.fc1, num_outputs=self.fc2, dropout_rate=self.dropout_rate,use_xavier=True, layers_collection=self.layers_collection, weights_collection=self.weights_collection, name='fc2')
        self.fc3_layer, self.weight_fc3 = utils.new_fc_layer(input=self.fc2_layer, num_inputs=self.fc2, num_outputs=self.fc3, use_relu=False, use_dropout=False,use_xavier=True, layers_collection=self.layers_collection, weights_collection=self.weights_collection, name='fc3')

        with tf.name_scope('loss'):
            # Build loss function
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_true, logits=self.fc3_layer)
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
            self.y_pred = tf.nn.softmax(self.fc3_layer)
            self.y_pred_cls = tf.argmax(self.y_pred, axis=1)
            self.correct_prediction = tf.equal(self.y_true_cls, self.y_pred_cls)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)
        
        print('-'*32)
        for layer in self.layers_collection:
            print(layer)
        print('-'*32)