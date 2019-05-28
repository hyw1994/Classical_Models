''' This is the VGG model from the original paper for study purpose.'''
import tensorflow as tf
import csv
import matplotlib.pyplot as plt
import numpy as np
from . import utils

# VGG network architecture.
# Initialization: 
# Weights: Xavier, 0 mean, 0.01 variation, .
# Bias: 0.

# [FILTER SHAPE], [OUTPUT SHAPE]
# []              [224, 224, 3]   INPUT LAYER: Subtracting the mean RGB value for training set. Remember the mean values for evaluation.
# [64, 3, 3]      [224, 224, 64]  CONV1_1 LAYER: 64 filters with size [3, 3] and stride=1, pad=1, activation=ReLU.
# [64, 3, 3]      [224, 224, 64]  CONV1_2 LAYER: 64 filters with size [3, 3] and stride=1, pad=1, activation=ReLU.

# [2, 2]          [112, 112, 64]  MAX POOLING1 LAYER: size=2, stride=2.

# [128, 3, 3]     [112, 112, 64]  CONV2_1 LAYER: 128 filters with size [3, 3] and stride=1, pad=1, activation=ReLU.
# [128, 3, 3]     [112, 112, 64]  CONV2_2 LAYER: 128 filters with size [3, 3] and stride=1, pad=1, activation=ReLU.

# [2, 2]          [56, 56, 64]    MAX POOLING2 LAYER: size=2, stride=2.

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

# [4096]          [4096]          FC1: 4096 neurons, ReLU, dropout=0.5.
# [4096]          [4096]          FC2: 4096 neurons, ReLU, dropout=0.5.
# [1000]          [1000]          FC3: 1000 neurons, Softmax. change it to the categories size while using.

class VGG():
    def __init__(self, num_classes):
        super().__init__()
        # Specify the network type, VGG16 or VGG19.
        self.num_classes = num_classes
        # Input image.
        self.x_image = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
        self.y_true_cls = tf.placeholder(tf.int64, shape=[None])
        
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

    def build(self, type='VGG16'):
        self.model_type = type
        print("The model in use is: {}".format(self.model_type))
        self.y_true = tf.one_hot(self.y_true_cls, depth=self.num_classes, axis=1)
        # Build conv1 layer.
        self.conv1_1_layer = utils.new_conv_layer(input=self.x_image,
                                                  num_input_channels=3,
                                                  paddings=self.layer_info.get('paddings'),
                                                  filter_size=self.layer_info.get('conv_size'),
                                                  stride=self.layer_info.get('conv_stride'),
                                                  num_filters=self.conv1.get('num_filters'),
                                                  name='conv1_1')
        
        self.conv1_2_layer = utils.new_conv_layer(input=self.conv1_1_layer,
                                                  num_input_channels=self.conv1.get('num_filters'),
                                                  paddings=self.layer_info.get('paddings'),
                                                  filter_size=self.layer_info.get('conv_size'),
                                                  stride=self.layer_info.get('conv_stride'),
                                                  num_filters=self.conv1.get('num_filters'),
                                                  name='conv1_2')
        
        self.pool1_layer = utils.new_pooling_layer(input=self.conv1_2_layer, 
                                                   pool_size=self.layer_info.get('pool_size'),
                                                   pool_stride=self.layer_info.get('pool_stride'),
                                                   name='pool1')

        # Build conv2 layer.
        self.conv2_1_layer = utils.new_conv_layer(input=self.pool1_layer,
                                                  num_input_channels=self.conv1.get('num_filters'),
                                                  paddings=self.layer_info.get('paddings'),
                                                  filter_size=self.layer_info.get('conv_size'),
                                                  stride=self.layer_info.get('conv_stride'),
                                                  num_filters=self.conv2.get('num_filters'),
                                                  name='conv2_1')
        
        self.conv2_2_layer = utils.new_conv_layer(input=self.conv2_1_layer,
                                                  num_input_channels=self.conv2.get('num_filters'),
                                                  paddings=self.layer_info.get('paddings'),
                                                  filter_size=self.layer_info.get('conv_size'),
                                                  stride=self.layer_info.get('conv_stride'),
                                                  num_filters=self.conv2.get('num_filters'),
                                                  name='conv2_2')

        self.pool2_layer = utils.new_pooling_layer(input=self.conv2_2_layer, 
                                                   pool_size=self.layer_info.get('pool_size'),
                                                   pool_stride=self.layer_info.get('pool_stride'),
                                                   name='pool2')

        # Build conv3 layer.
        self.conv3_1_layer = utils.new_conv_layer(input=self.conv2_2_layer,
                                                  num_input_channels=self.conv2.get('num_filters'),
                                                  paddings=self.layer_info.get('paddings'),
                                                  filter_size=self.layer_info.get('conv_size'),
                                                  stride=self.layer_info.get('conv_stride'),
                                                  num_filters=self.conv3.get('num_filters'),
                                                  name='conv3_1')
        
        self.conv3_2_layer = utils.new_conv_layer(input=self.conv3_1_layer,
                                                  num_input_channels=self.conv3.get('num_filters'),
                                                  paddings=self.layer_info.get('paddings'),
                                                  filter_size=self.layer_info.get('conv_size'),
                                                  stride=self.layer_info.get('conv_stride'),
                                                  num_filters=self.conv3.get('num_filters'),
                                                  name='conv3_2')
                                                  
        self.conv3_3_layer = utils.new_conv_layer(input=self.conv3_2_layer,
                                                  num_input_channels=self.conv3.get('num_filters'),
                                                  paddings=self.layer_info.get('paddings'),
                                                  filter_size=self.layer_info.get('conv_size'),
                                                  stride=self.layer_info.get('conv_stride'),
                                                  num_filters=self.conv3.get('num_filters'),
                                                  name='conv3_3')

        self.conv3_output = self.conv3_3_layer

        if(self.model_type == 'VGG19' or self.model_type == 'vgg19'):
            self.conv3_4_layer = utils.new_conv_layer(input=self.conv3_3_layer,
                                                      num_input_channels=self.conv3.get('num_filters'),
                                                      paddings=self.layer_info.get('paddings'),
                                                      filter_size=self.layer_info.get('conv_size'),
                                                      stride=self.layer_info.get('conv_stride'),
                                                      num_filters=self.conv3.get('num_filters'),
                                                      name='conv3_4')
            self.conv3_output = self.conv3_4_layer

        self.pool3_layer = utils.new_pooling_layer(input=self.conv3_output, 
                                                   pool_size=self.layer_info.get('pool_size'),
                                                   pool_stride=self.layer_info.get('pool_stride'),
                                                   name='pool3')

        # Build conv4 layer.
        self.conv4_1_layer = utils.new_conv_layer(input=self.pool3_layer,
                                                  num_input_channels=self.conv3.get('num_filters'),
                                                  paddings=self.layer_info.get('paddings'),
                                                  filter_size=self.layer_info.get('conv_size'),
                                                  stride=self.layer_info.get('conv_stride'),
                                                  num_filters=self.conv4.get('num_filters'),
                                                  name='conv4_1')
        
        self.conv4_2_layer = utils.new_conv_layer(input=self.conv4_1_layer,
                                                  num_input_channels=self.conv4.get('num_filters'),
                                                  paddings=self.layer_info.get('paddings'),
                                                  filter_size=self.layer_info.get('conv_size'),
                                                  stride=self.layer_info.get('conv_stride'),
                                                  num_filters=self.conv4.get('num_filters'),
                                                  name='conv4_2')
                                                  
        self.conv4_3_layer = utils.new_conv_layer(input=self.conv4_2_layer,
                                                  num_input_channels=self.conv4.get('num_filters'),
                                                  paddings=self.layer_info.get('paddings'),
                                                  filter_size=self.layer_info.get('conv_size'),
                                                  stride=self.layer_info.get('conv_stride'),
                                                  num_filters=self.conv4.get('num_filters'),
                                                  name='conv4_3')

        self.conv4_output = self.conv4_3_layer

        if(self.model_type == 'VGG19' or self.model_type == 'vgg19'):
            self.conv4_4_layer = utils.new_conv_layer(input=self.conv4_3_layer,
                                                      num_input_channels=self.conv4.get('num_filters'),
                                                      paddings=self.layer_info.get('paddings'),
                                                      filter_size=self.layer_info.get('conv_size'),
                                                      stride=self.layer_info.get('conv_stride'),
                                                      num_filters=self.conv4.get('num_filters'),
                                                      name='conv4_4')
            self.conv4_output = self.conv4_4_layer

        self.pool4_layer = utils.new_pooling_layer(input=self.conv4_output, 
                                                   pool_size=self.layer_info.get('pool_size'),
                                                   pool_stride=self.layer_info.get('pool_stride'),
                                                   name='pool4')
        
        # Build conv5 layer.
        self.conv5_1_layer = utils.new_conv_layer(input=self.pool4_layer,
                                                  num_input_channels=self.conv4.get('num_filters'),
                                                  paddings=self.layer_info.get('paddings'),
                                                  filter_size=self.layer_info.get('conv_size'),
                                                  stride=self.layer_info.get('conv_stride'),
                                                  num_filters=self.conv5.get('num_filters'),
                                                  name='conv5_1')
        
        self.conv5_2_layer = utils.new_conv_layer(input=self.conv5_1_layer,
                                                  num_input_channels=self.conv5.get('num_filters'),
                                                  paddings=self.layer_info.get('paddings'),
                                                  filter_size=self.layer_info.get('conv_size'),
                                                  stride=self.layer_info.get('conv_stride'),
                                                  num_filters=self.conv5.get('num_filters'),
                                                  name='conv5_2')
                                                  
        self.conv5_3_layer = utils.new_conv_layer(input=self.conv5_2_layer,
                                                  num_input_channels=self.conv5.get('num_filters'),
                                                  paddings=self.layer_info.get('paddings'),
                                                  filter_size=self.layer_info.get('conv_size'),
                                                  stride=self.layer_info.get('conv_stride'),
                                                  num_filters=self.conv5.get('num_filters'),
                                                  name='conv5_3')

        self.conv5_output = self.conv5_3_layer

        if(self.model_type == 'VGG19' or self.model_type == 'vgg19'):
            self.conv5_4_layer = utils.new_conv_layer(input=self.conv5_3_layer,
                                                      num_input_channels=self.conv5.get('num_filters'),
                                                      paddings=self.layer_info.get('paddings'),
                                                      filter_size=self.layer_info.get('conv_size'),
                                                      stride=self.layer_info.get('conv_stride'),
                                                      num_filters=self.conv5.get('num_filters'),
                                                      name='conv5_4')
            self.conv5_output = self.conv5_4_layer

        self.pool5_layer = utils.new_pooling_layer(input=self.conv5_output, 
                                                   pool_size=self.layer_info.get('pool_size'),
                                                   pool_stride=self.layer_info.get('pool_stride'),
                                                   name='pool5')