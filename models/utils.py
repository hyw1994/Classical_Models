import tensorflow as tf
import pathlib
import scipy.io as scio
import random

def new_weights(shape, name, use_xavier=False):
    # Create tf.Variable for filters.
    if(use_xavier):
        return tf.Variable(tf.glorot_uniform_initializer()(shape), name=name+'-W')
    else:
        return tf.Variable(tf.random.truncated_normal(shape, stddev=0.01), name=name+'-W')

def new_biases(length, value, name):
    # Create tf.Variable for bias.
    return tf.Variable(tf.constant(value, shape=[length]), name=name+'-b')

def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   stride,             # stride size of the filter.
                   paddings=0,           # padding shape.
                   use_bias = True,    # Use bias or not.
                   bias_value = 0.0,   # How to initialize the bias.
                   use_norm = False,   # Use norm or not.
                   use_pooling=False,  # Use 3x3 max-pooling by default.
                   pool_size = 3,
                   pool_stride = 2,
                   name='conv',
                   layers_collection = [],
                   weights_collection = [],
                   use_xavier=False): 

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    with tf.name_scope(name):
        filter_shape = [filter_size, filter_size, num_input_channels, num_filters]
        
        # Create padding constant.
        # Pad the layer
        layer = input
        if(paddings > 0):
            paddings = tf.constant([[0, 0],[paddings, paddings], [paddings, paddings], [0, 0]])
            layer = tf.pad(tensor=layer, 
                            paddings=paddings, 
                            mode='CONSTANT', name=name + '-padding')

        # Create new weights aka. filters with the given shape.
        weights = new_weights(shape=filter_shape, name=name, use_xavier=use_xavier)
            
        # Do the convolotion job with different parameters settings.
        layer = tf.nn.conv2d(input=layer,
                            filter=weights,
                            strides=[1, stride, stride, 1],
                            padding='VALID', name=name + '-conv')

        # Add the biases to the results of the convolution.
        # A bias-value is added to each filter-channel.
        if(use_bias):
            biases = new_biases(length=num_filters, value=bias_value, name=name)
            layer = tf.add(layer, biases)

        # ReLU activation is use.
        layer = tf.nn.relu(layer, name=name + '-relu')

        # Local_response_normalization is used.
        if(use_norm):
            layer = tf.nn.local_response_normalization(input=layer,
                                                    depth_radius=5,
                                                    bias=2.0,
                                                    alpha=10**-4,
                                                    beta=0.75, name=name + '-lr-norm')

        # Use pooling to down-sample the image resolution.
        if use_pooling:
            # This is an overlapping 3x3 max-pooling, with stride 2.
            layer = tf.nn.max_pool(value=layer,
                                ksize=[1, pool_size, pool_size, 1],
                                strides=[1, pool_stride, pool_stride, 1],
                                padding='VALID', name=name+'-pooling')

        # summary
        tf.summary.histogram(name+'weights', weights)
        tf.summary.histogram(name+'biases', biases)
        tf.summary.histogram(name+'activations', layer)

        # We return both the resulting layer and the filter-weights
        # because we will plot the weights later.
        layers_collection.append(layer)
        weights_collection.append(weights)
        return layer, weights

def new_fc_layer(input,
                 num_inputs,
                 num_outputs,
                 use_relu=True,
                 use_dropout=True, 
                 dropout_rate=tf.placeholder_with_default(0.5, shape=()),
                 name='fc',
                 layers_collection=[],
                 weights_collection=[],
                 use_xavier=False):
    with tf.name_scope(name):
        # Create new weights and biases.
        weights = new_weights(shape=[num_inputs, num_outputs], name=name, use_xavier=use_xavier)
        biases = new_biases(length=num_outputs, value=1.0, name=name)
        # Do the matrix multiple and get the output neurons.
        layer = tf.add(tf.matmul(input, weights), biases, name=name + '-add')
        # Add the ReLU activation here.
        if use_relu:
            layer = tf.nn.relu(layer, name=name+'-relu')
        
        if use_dropout:
            layer = tf.nn.dropout(layer, rate=dropout_rate, name=name+'-dropout')

        # summary
        tf.summary.histogram(name+'weights', weights)
        tf.summary.histogram(name+'biases', biases)
        tf.summary.histogram(name+'activations', layer)

        layers_collection.append(layer)
        weights_collection.append(weights)
        
        return layer, weights

def new_pooling_layer(input, pool_size, pool_stride, name, layers_collection=[]):
    with tf.name_scope(name):
        layer = tf.nn.max_pool(value=input, 
                            ksize=[1, pool_size, pool_size, 1],
                            strides=[1, pool_stride, pool_stride, 1],
                            padding='VALID', name=name+'-pooling')
        # summary
        tf.summary.histogram(name+'pooling', layer)

        layers_collection.append(layer)
        return layer

def flatten_layer(input, layers_collection=[], name='flatten'):
    with tf.name_scope(name):
        # This function is used to flat the final CONV layer to connect it to FC layer.
        layer_shape = input.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer_flatten = tf.reshape(input, [-1, num_features], name=name + '-flatten')
        layers_collection.append(layer_flatten)
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
    return image 

def load_and_preprocess_image(path, labels):
    raw_image = tf.read_file(path)

    return preprocess_image(raw_image), labels

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

def split_train_cv_test_set(images, labels):
    train_pair = (images[0: 6142], labels[0: 6142])
    cv_pair = (images[6142: 7143], labels[6142: 7143])
    test_pair = (images[6142: 7143], labels[6142: 7143])
    return train_pair, cv_pair, train_pair

def load_data(image_root, label_root):
    all_image_paths, all_image_labels = load_labels_and_image_path(image_root, label_root)
    train_pair, cv_pair, test_pair = split_train_cv_test_set(all_image_paths, all_image_labels)

    train_path_ds = tf.data.Dataset.from_tensor_slices(train_pair)
    cv_path_ds = tf.data.Dataset.from_tensor_slices(cv_pair)
    test_path_ds = tf.data.Dataset.from_tensor_slices(test_pair)

    train_ds = train_path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    cv_ds = cv_path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return train_ds, cv_ds, test_ds

def pre_process_ds_images(image, label, size=227):
    """ reshape the image to [227, 227] and one-hot the labels to fit the network."""
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_images(images=image, size=[size, size])
    image = image/255.0
    return image, label

def prepare_train_ds(train_ds, BATCH_SIZE, INPUT_SIZE, image_size=227):
    """We shuffle and batch the training samples to make the training process work better."""
    # Resize the input image size.
    train_ds = train_ds.map(lambda image, label: pre_process_ds_images(image, label, size=image_size)) 

    # We prepare the shuffle buffer to be the same size as the whole size of the input sample to make it shuffle globally.
    train_ds = train_ds.shuffle(buffer_size=INPUT_SIZE)
    train_ds = train_ds.repeat()
    train_ds = train_ds.batch(BATCH_SIZE)
    # Prefetch the data to accerlate the training process. 
    train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_ds