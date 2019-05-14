import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from sklearn.metrics import confusion_matrix
import time

# This is the AlexNet model introduced by the paper "ImageNet Classification with Deep Convolutional Neural Networks".

# The network contains five CONV layer and three FC layer. The detailed layer information of the modele is: 
# [227*227*3] INPUT LAYER: 1.2 million training images, 50,000 validation inages, 150,000 testing images with size 227*227*3, 0 padding. 
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
# [1000] FC8: 1000 neurons

# Step 1: Load dataset from 102 category flower dataset 
data_root = tf.keras.utils.get_file('jpg', 'http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz', untar=True)
data_root = pathlib.Path(data_root)
print(data_root)

# Step 2: Define the network
# Input image
image_size = 227
image_channel = 3
num_classes = 1000

# Convolutional Layer 1.
conv1_params = {
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
conv2_params = {
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
conv3_params = {
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
conv4_params = {
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
conv5_params = {
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
fc6_size = 4096             
fc7_size = 4096
fc8_size = 1000

# Step 3: Define the helper function 
def new_weights(shape):
    # Create tf.Variable for filters.
    return tf.Variable(tf.random.truncated_normal(shape, stddev=0.01))

def new_biases(length, value):
    # Create tf.Variable for bias.
    return tf.Variable(tf.constant(value, shape=[length]))

def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   stride,             # stride size of the filter.
                   paddings=0,           # padding shape.
                   use_bias = True,    # Use bias or not.
                   bias_value = 0.0,   # How to initialize the bias.
                   use_norm = False,   # Use norm or not.
                   use_pooling=False): # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    filter_shape = [filter_size, filter_size, num_input_channels, num_filters]
    
    # Create padding constant.
    # Pad the layer
    paddings = tf.constant([[0, 0],[paddings, paddings], [paddings, paddings], [0, 0]])
    layer = tf.pad(tensor=input, 
                   paddings=paddings, 
                   mode='CONSTANT')

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=filter_shape)
        

    # Do the convolotion job with different parameters settings.
    layer = tf.nn.conv2d(input=layer,
                         filter=weights,
                         strides=[1, stride, stride, 1],
                         padding='VALID')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    if(use_bias):
        biases = new_biases(length=num_filters, value=bias_value)
        layer += biases

    # ReLU activation is use.
    layer = tf.nn.relu(layer)

    # Local_response_normalization is used.
    if(use_norm):
        layer = tf.nn.local_response_normalization(input=layer,
                                                   depth_radius=5,
                                                   bias=2.0,
                                                   alpha=10**-4,
                                                   beta=0.75)

    # Use pooling to down-sample the image resolution.
    if use_pooling:
        # This is an overlapping 3x3 max-pooling, with stride 2.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1],
                               padding='VALID')

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights

def new_fc_layer(input,
                 num_inputs,
                 num_outputs,
                 use_relu=True,
                 use_dropout=False):
    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs, value=1.0)
    # Do the matrix multiple and get the output neurons.
    layer = tf.matmul(input, weights) + biases
    # Add the ReLU activation here.
    if use_relu:
        layer = tf.nn.relu(layer)
    
    if use_dropout:
        layer = tf.nn.dropout(layer, rate=0.5)

    return layer

def flatten_layer(layer):
    # This function is used to flat the final CONV layer to connect it to FC layer.
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flatten = tf.reshape(layer, [-1, num_features])

    return layer_flatten, num_features

    

# Step 4: Build network
# Input layer and Output classes
x_image = tf.placeholder(tf.float32, shape=[None, image_size, image_size, image_channel])
y_train_true = tf.placeholder(tf.float32, shape=[None, num_classes], name="y_train_true")
y_train_cls = tf.argmax(y_train_true, axis=1)

# CONV1 layer
conv1_layer, weight1 = new_conv_layer(input=x_image,
                                num_input_channels=image_channel,
                                filter_size=conv1_params.get('filter_size'),
                                num_filters=conv1_params.get('num_filters'),
                                stride=conv1_params.get('stride'),
                                paddings=conv1_params.get('paddings'),
                                use_bias=conv1_params.get('use_bias'),
                                bias_value=conv1_params.get('bias_value'),
                                use_norm=conv1_params.get('use_norm'),
                                use_pooling=conv1_params.get('use_pooling'))

# CONV2 layer
conv2_layer, weight2 = new_conv_layer(input=conv1_layer,
                                num_input_channels=conv1_params.get('num_filters'),
                                filter_size=conv2_params.get('filter_size'),
                                num_filters=conv2_params.get('num_filters'),
                                stride=conv2_params.get('stride'),
                                paddings=conv2_params.get('paddings'),
                                use_bias=conv2_params.get('use_bias'),
                                bias_value=conv2_params.get('bias_value'),
                                use_norm=conv2_params.get('use_norm'),
                                use_pooling=conv2_params.get('use_pooling'))

# CONV3 layer
conv3_layer, weight3 = new_conv_layer(input=conv2_layer,
                                num_input_channels=conv2_params.get('num_filters'),
                                filter_size=conv3_params.get('filter_size'),
                                num_filters=conv3_params.get('num_filters'),
                                stride=conv3_params.get('stride'),
                                paddings=conv3_params.get('paddings'),
                                use_bias=conv3_params.get('use_bias'),
                                bias_value=conv3_params.get('bias_value'),
                                use_norm=conv3_params.get('use_norm'),
                                use_pooling=conv3_params.get('use_pooling'))

# CONV4 layer
conv4_layer, weight4 = new_conv_layer(input=conv3_layer,
                                num_input_channels=conv3_params.get('num_filters'),
                                filter_size=conv4_params.get('filter_size'),
                                num_filters=conv4_params.get('num_filters'),
                                stride=conv4_params.get('stride'),
                                paddings=conv4_params.get('paddings'),
                                use_bias=conv4_params.get('use_bias'),
                                bias_value=conv4_params.get('bias_value'),
                                use_norm=conv4_params.get('use_norm'),
                                use_pooling=conv4_params.get('use_pooling'))

# CONV5 layer
conv5_layer, weight5 = new_conv_layer(input=conv4_layer,
                                num_input_channels=conv4_params.get('num_filters'),
                                filter_size=conv5_params.get('filter_size'),
                                num_filters=conv5_params.get('num_filters'),
                                stride=conv5_params.get('stride'),
                                paddings=conv5_params.get('paddings'),
                                use_bias=conv5_params.get('use_bias'),
                                bias_value=conv5_params.get('bias_value'),
                                use_norm=conv5_params.get('use_norm'),
                                use_pooling=conv5_params.get('use_pooling'))

# Flatten the last CONV layer
flattened_layer, num_features = flatten_layer(conv5_layer)

# FC layers
fc6_layer = new_fc_layer(flattened_layer, num_inputs=num_features, num_outputs=fc6_size, use_dropout=True)
fc7_layer = new_fc_layer(fc6_layer, num_inputs=fc6_size, num_outputs=fc7_size, use_dropout=True)
fc8_layer = new_fc_layer(fc7_layer, num_inputs=fc7_size, num_outputs=fc8_size)

print(conv1_layer)
print(conv2_layer)
print(conv3_layer)
print(conv4_layer)
print(conv5_layer)
print(flattened_layer)
print(fc6_layer)
print(fc7_layer)
print(fc8_layer)