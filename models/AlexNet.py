'''This is the AlexNet model introduced by the paper "ImageNet Classification with Deep Convolutional Neural Networks".'''
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from . import utils
from .model import model

# The network contains five CONV layer and three FC layer. The detailed layer information of the modele is: 
# [227*227*3] INPUT LAYER: reshape the image and divide the image by 255.0 to rescale the distribution to [0, 1]. 
# [55*55*96] CONV1 LAYER: 96 11*11*3, stride 4, pad 0, ReLU activation.
# [55*55*96] RESPONSE NORM LAYER;
# [27*27*96] MAX POOLING1 LAYER: 3*3, stride 2.
# [27*27*256] CONV2 LAYER: 256 5*5*96, stride 4, pad 2.
# [27*27*256] RESPONSE NORM LAYER;
# [13*13*256] MAX POOLING2 LAYER: 3*3, stride 2.
# [13*13*384] CONV3 LAYER: 384 3*3*256, stride 1, pad 1.
# [13*13*384] CONV4 LAYER: 384 3*3*384, stride 1, pad 1.
# [13*13*256] CONV5 LAYER: 256 3*3*384, stride 1, pad 1.
# [6*6*256] MAX POOLING3 LAYER: 3*3, stride 2.
# [4096] FC6: 4096 neurons
# [4096] FC7: 4096 neurons
# [1000] FC8: 1000 neurons, change it to the categories size while using.

class AlexNet(model):
    def __init__(self, dataset_name, num_classes):
        super().__init__(model_name='AlexNet', dataset_name=dataset_name)
        # Step 1: Define the network
        # Input image.
        self.num_classes = num_classes
        self.x_image = tf.placeholder(tf.float32, shape=[None, 227, 227, 3])
        self.y_true_cls = tf.placeholder(tf.int64, shape=[None])

        # Convolutional Layer 1.
        self.conv1_params = {
            "filter_size": 11,
            "num_filters": 96,
            "stride": 4,
            "paddings": 0,
            "use_bias": True,
            "bias_value": 0.0,
            "use_norm": True,
            "use_pooling": True
        }

        # Convolutional Layer 2.
        self.conv2_params = {
            "filter_size": 5,
            "num_filters": 256,
            "stride": 1,
            "paddings": 2,
            "use_bias": True,
            "bias_value": 1.0,
            "use_norm": True,
            "use_pooling": True
        }

        # Convolutional Layer 3.
        self.conv3_params = {
            "filter_size": 3,
            "num_filters": 384,
            "stride": 1,
            "paddings": 1,
            "use_bias": True,
            "bias_value": 0.0,
            "use_norm": False,
            "use_pooling": False
        }

        # Convolutional Layer 4.
        self.conv4_params = {
            "filter_size": 3,
            "num_filters": 384,
            "stride": 1,
            "paddings": 1,
            "use_bias": True,
            "bias_value": 1.0,
            "use_norm": False,
            "use_pooling": False
        }

        # Convolutional Layer 5.
        self.conv5_params = {
            "filter_size": 3,
            "num_filters": 256,
            "stride": 1,
            "paddings": 1,
            "use_bias": True,
            "bias_value": 1.0,
            "use_norm": False,
            "use_pooling": True
        }

        # Fully-connected layer, bias=1.0.
        self.fc6_size = 4096             
        self.fc7_size = 4096
        self.fc8_size = self.num_classes

    def build(self):
        self.y_true = tf.one_hot(self.y_true_cls, depth=self.num_classes, axis=1)
        # Conv1: 96 11*11 filters, stride 4, relu 0, local response normalization, max pooing with z=3, s=2.
        self.conv1_layer, self.weight1 = utils.new_conv_layer(input=self.x_image,
                                                              num_input_channels=3,
                                                              filter_size=self.conv1_params.get('filter_size'),
                                                              num_filters=self.conv1_params.get('num_filters'),
                                                              stride=self.conv1_params.get('stride'),
                                                              paddings=self.conv1_params.get('paddings'),
                                                              use_bias=self.conv1_params.get('use_bias'),
                                                              bias_value=self.conv1_params.get('bias_value'),
                                                              use_norm=self.conv1_params.get('use_norm'),
                                                              use_pooling=self.conv1_params.get('use_pooling'), 
                                                              name='conv1')

        # Conv2: pad 2 256 5*5 filters, stride 1, relu 1, local response normalization, max pooing with z=3, s=2.
        self.conv2_layer, self.weight2 = utils.new_conv_layer(input=self.conv1_layer,
                                                              num_input_channels=self.conv1_params.get('num_filters'),
                                                              filter_size=self.conv2_params.get('filter_size'),
                                                              num_filters=self.conv2_params.get('num_filters'),
                                                              stride=self.conv2_params.get('stride'),
                                                              paddings=self.conv2_params.get('paddings'),
                                                              use_bias=self.conv2_params.get('use_bias'),
                                                              bias_value=self.conv2_params.get('bias_value'),
                                                              use_norm=self.conv2_params.get('use_norm'),
                                                              use_pooling=self.conv2_params.get('use_pooling'),
                                                              name='conv2')

        # Conv3: pad 1 384 3*3 filters, stride 1, relu 0.
        self.conv3_layer, self.weight3 = utils.new_conv_layer(input=self.conv2_layer,
                                                              num_input_channels=self.conv2_params.get('num_filters'),
                                                              filter_size=self.conv3_params.get('filter_size'),
                                                              num_filters=self.conv3_params.get('num_filters'),
                                                              stride=self.conv3_params.get('stride'),
                                                              paddings=self.conv3_params.get('paddings'),
                                                              use_bias=self.conv3_params.get('use_bias'),
                                                              bias_value=self.conv3_params.get('bias_value'),
                                                              use_norm=self.conv3_params.get('use_norm'),
                                                              use_pooling=self.conv3_params.get('use_pooling'),
                                                              name='conv3')

        # Conv4: pad1 384 3*3 filters, stride 1,relu 0.
        self.conv4_layer, self.weight4 = utils.new_conv_layer(input=self.conv3_layer,
                                                              num_input_channels=self.conv3_params.get('num_filters'),
                                                              filter_size=self.conv4_params.get('filter_size'),
                                                              num_filters=self.conv4_params.get('num_filters'),
                                                              stride=self.conv4_params.get('stride'),
                                                              paddings=self.conv4_params.get('paddings'),
                                                              use_bias=self.conv4_params.get('use_bias'),
                                                              bias_value=self.conv4_params.get('bias_value'),
                                                              use_norm=self.conv4_params.get('use_norm'),
                                                              use_pooling=self.conv4_params.get('use_pooling'),
                                                              name='conv4')

        # Conv5: pad1 384 3*3 filters, stride 1,relu 1, max pooing with z=3, s=2.
        self.conv5_layer, self.weight5 = utils.new_conv_layer(input=self.conv4_layer,
                                                              num_input_channels=self.conv4_params.get('num_filters'),
                                                              filter_size=self.conv5_params.get('filter_size'),
                                                              num_filters=self.conv5_params.get('num_filters'),
                                                              stride=self.conv5_params.get('stride'),
                                                              paddings=self.conv5_params.get('paddings'),
                                                              use_bias=self.conv5_params.get('use_bias'),
                                                              bias_value=self.conv5_params.get('bias_value'),
                                                              use_norm=self.conv5_params.get('use_norm'),
                                                              use_pooling=self.conv5_params.get('use_pooling'),
                                                              name='conv5')

        # Flatten the last CONV layer
        self.flattened_layer, self.num_features = utils.flatten_layer(input=self.conv5_layer, name='flatten')

        # FC layers, 4096, 4096, 100
        self.dropout_rate = tf.placeholder_with_default(0.5, shape=())
        self.fc6_layer, self.weight_fc6 = utils.new_fc_layer(self.flattened_layer, num_inputs=self.num_features, num_outputs=self.fc6_size, dropout_rate=self.dropout_rate, name='fc6')
        self.fc7_layer, self.weight_fc7 = utils.new_fc_layer(self.fc6_layer, num_inputs=self.fc6_size, num_outputs=self.fc7_size, dropout_rate=self.dropout_rate, name='fc7')
        self.fc8_layer, self.weight_fc8 = utils.new_fc_layer(self.fc7_layer, num_inputs=self.fc7_size, num_outputs=self.fc8_size, use_relu=False, use_dropout=False, name='fc8')
        
        with tf.name_scope('loss'):
            # loss function
            self.y_pred = tf.nn.softmax(self.fc8_layer)
            self.y_pred_cls = tf.argmax(self.y_pred, axis=1)
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.fc8_layer, labels=self.y_true)
            self.cost = tf.reduce_mean(self.cross_entropy)
            
            tf.summary.scalar('cross_entropy_loss', self.cost)

        with tf.name_scope('optimizer'):
            # optimizer
            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.cost)

        with tf.name_scope('accuracy'):
            # Performance Measured
            self.correct_prediction = tf.equal(self.y_true_cls, self.y_pred_cls)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        print('-' * 32)
        print("The model has been built successfully!")
        print("The model structure is:")
        print(self.conv1_layer)
        print(self.conv2_layer)
        print(self.conv3_layer)
        print(self.conv4_layer)
        print(self.conv5_layer)
        print(self.flattened_layer)
        print(self.fc6_layer)
        print(self.fc7_layer)
        print(self.fc8_layer)
        print('-' * 32)