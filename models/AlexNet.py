import tensorflow as tf
from . import utils

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

class AlexNet:
    def __init__(self):
        super().__init__()

        # Step 1: Define the network
        # Input image.
        self.image_size = 227
        self.image_channel = 3
        self.num_classes = 102

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

        # Step 4: Build network
        # Input layer and Output classes
        self.x_image = tf.placeholder(tf.float32, shape=[None, self.image_size, self.image_size, self.image_channel])
        self.y_train_true = tf.placeholder(tf.float32, shape=[None, self.num_classes], name="y_train_true")
        self.y_train_cls = tf.argmax(self.y_train_true, axis=1)

    def build(self):
        # CONV1 layer
        self.conv1_layer, self.weight1 = utils.new_conv_layer(input=self.x_image,
                                        num_input_channels=self.image_channel,
                                        filter_size=self.conv1_params.get('filter_size'),
                                        num_filters=self.conv1_params.get('num_filters'),
                                        stride=self.conv1_params.get('stride'),
                                        paddings=self.conv1_params.get('paddings'),
                                        use_bias=self.conv1_params.get('use_bias'),
                                        bias_value=self.conv1_params.get('bias_value'),
                                        use_norm=self.conv1_params.get('use_norm'),
                                        use_pooling=self.conv1_params.get('use_pooling'))

        # CONV2 layer
        self.conv2_layer, self.weight2 = utils.new_conv_layer(input=self.conv1_layer,
                                        num_input_channels=self.conv1_params.get('num_filters'),
                                        filter_size=self.conv2_params.get('filter_size'),
                                        num_filters=self.conv2_params.get('num_filters'),
                                        stride=self.conv2_params.get('stride'),
                                        paddings=self.conv2_params.get('paddings'),
                                        use_bias=self.conv2_params.get('use_bias'),
                                        bias_value=self.conv2_params.get('bias_value'),
                                        use_norm=self.conv2_params.get('use_norm'),
                                        use_pooling=self.conv2_params.get('use_pooling'))

        # CONV3 layer
        self.conv3_layer, self.weight3 = utils.new_conv_layer(input=self.conv2_layer,
                                        num_input_channels=self.conv2_params.get('num_filters'),
                                        filter_size=self.conv3_params.get('filter_size'),
                                        num_filters=self.conv3_params.get('num_filters'),
                                        stride=self.conv3_params.get('stride'),
                                        paddings=self.conv3_params.get('paddings'),
                                        use_bias=self.conv3_params.get('use_bias'),
                                        bias_value=self.conv3_params.get('bias_value'),
                                        use_norm=self.conv3_params.get('use_norm'),
                                        use_pooling=self.conv3_params.get('use_pooling'))

        # CONV4 layer
        self.conv4_layer, self.weight4 = utils.new_conv_layer(input=self.conv3_layer,
                                        num_input_channels=self.conv3_params.get('num_filters'),
                                        filter_size=self.conv4_params.get('filter_size'),
                                        num_filters=self.conv4_params.get('num_filters'),
                                        stride=self.conv4_params.get('stride'),
                                        paddings=self.conv4_params.get('paddings'),
                                        use_bias=self.conv4_params.get('use_bias'),
                                        bias_value=self.conv4_params.get('bias_value'),
                                        use_norm=self.conv4_params.get('use_norm'),
                                        use_pooling=self.conv4_params.get('use_pooling'))

        # CONV5 layer
        self.conv5_layer, self.weight5 = utils.new_conv_layer(input=self.conv4_layer,
                                        num_input_channels=self.conv4_params.get('num_filters'),
                                        filter_size=self.conv5_params.get('filter_size'),
                                        num_filters=self.conv5_params.get('num_filters'),
                                        stride=self.conv5_params.get('stride'),
                                        paddings=self.conv5_params.get('paddings'),
                                        use_bias=self.conv5_params.get('use_bias'),
                                        bias_value=self.conv5_params.get('bias_value'),
                                        use_norm=self.conv5_params.get('use_norm'),
                                        use_pooling=self.conv5_params.get('use_pooling'))

        # Flatten the last CONV layer
        self.flattened_layer, self.num_features = utils.flatten_layer(self.conv5_layer)

        # FC layers
        self.fc6_layer = utils.new_fc_layer(self.flattened_layer, num_inputs=self.num_features, num_outputs=self.fc6_size, use_dropout=True)
        self.fc7_layer = utils.new_fc_layer(self.fc6_layer, num_inputs=self.fc6_size, num_outputs=self.fc7_size, use_dropout=True)
        self.fc8_layer = utils.new_fc_layer(self.fc7_layer, num_inputs=self.fc7_size, num_outputs=self.fc8_size)

        # optimizer
        self.y_train_pred = tf.nn.softmax(self.fc8_layer)
        self.y_train_pred_cls = tf.argmax(self.y_train_pred, axis=1)

        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.fc8_layer, labels=self.y_train_true)
        self.cost = tf.reduce_mean(self.cross_entropy)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.cost)

        # Performance Measured
        self.correct_prediction = tf.equal(self.y_train_true, self.y_train_pred)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        print(self.conv1_layer)
        print(self.conv2_layer)
        print(self.conv3_layer)
        print(self.conv4_layer)
        print(self.conv5_layer)
        print(self.flattened_layer)
        print(self.fc6_layer)
        print(self.fc7_layer)
        print(self.fc8_layer)

    def train(self):
        
        return