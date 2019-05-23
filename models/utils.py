import tensorflow as tf
import pathlib
import scipy.io as scio
import random

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
        layer = tf.add(layer, biases)

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
    layer = tf.add(tf.matmul(input, weights), biases)
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

def download_images():
    label_root = tf.keras.utils.get_file('imagelabels.mat', 'http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat')
    image_root = tf.keras.utils.get_file('jpg', 'http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz', untar=True)
    label_root = pathlib.Path(label_root)
    image_root = pathlib.Path(image_root)

    print('*' * 32)
    print("The image set has been downloaded in the path: " + str(image_root))
    print("The label set has been downloaded in the path: " + str(label_root))
    print('*' * 32)
    
    return image_root, label_root

def get_image_index(image_path):
    image_index = image_path.split('_')[-1][:-4]
    return image_index

def preprocess_image(raw_image):
    image = tf.image.decode_jpeg(raw_image, channels=3)
    image_final = tf.image.resize_images(image, [227, 227])
    image_final = image_final / 255.0
    return image_final 

def load_and_preprocess_image(path):
    raw_image = tf.read_file(path)

    return preprocess_image(raw_image)

def load_labels_and_image_path(image_root, label_root):
    labels = scio.loadmat(str(label_root)).get('labels')[0]

    all_image_paths = list(image_root.glob('*.jpg'))
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)

    image_prefix = all_image_paths[0][:-9]
    image_index = [get_image_index(path) for path in all_image_paths]
    image_postfix = ".jpg"

    all_image_labels = [labels[int(index) - 1] for index in image_index]

    return all_image_paths, all_image_labels

def load_data(image_root, label_root):
    all_image_paths, all_image_labels = load_labels_and_image_path(image_root, label_root)
    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int32))

    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
    return image_label_ds