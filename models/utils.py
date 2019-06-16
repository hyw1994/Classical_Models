import tensorflow as tf
import keras
import pathlib
import numpy as np
import scipy.io as scio
import random
import os

def new_weights(shape, name, use_xavier=False, use_MSRA=False):
    # Create tf.Variable for filters.
    if(use_xavier):
        return tf.Variable(tf.glorot_uniform_initializer()(shape), name=name+'-W')
        # return tf.get_variable(name=name+'-W', shape=shape, initializer=tf.glorot_uniform_initializer())
    elif(use_MSRA):
        return tf.Variable(tf.contrib.layers.variance_scaling_initializer()(shape), name=name+'-W')
        # return tf.get_variable(name=name+'-W', shape=shape, initializer=tf.contrib.layers.variance_scaling_initializer())
    else:
        return tf.Variable(tf.truncated_normal_initializer(mean=0, stddev=0.01)(shape), name=name+'-W')
        # return tf.get_variable(name=name+'-W', shape=shape, initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))

def new_biases(length, value, name):
    # Create tf.Variable for bias.
    return tf.Variable(tf.initializers.constant(value=value)([length]), name=name+'-b')
    # return tf.get_variable(name=name+'-b', shape=length, initializer=tf.initializers.constant(value=value))

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
                   padding_mode='VALID',
                   layers_collection = [],
                   weights_collection = [],
                   use_xavier=False,
                   use_relu=True,
                   use_MSRA=False): 

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
        weights = new_weights(shape=filter_shape, name=name, use_xavier=use_xavier, use_MSRA=use_MSRA)
            
        # Do the convolotion job with different parameters settings.
        layer = tf.nn.conv2d(input=layer,
                            filter=weights,
                            strides=[1, stride, stride, 1],
                            padding=padding_mode, name=name + '-conv')

        # Add the biases to the results of the convolution.
        # A bias-value is added to each filter-channel.
        if(use_bias):
            biases = new_biases(length=num_filters, value=bias_value, name=name)
            layer = tf.add(layer, biases)

        # ReLU activation is use.
        if(use_relu):
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
                 bias_value = 1.0,
                 layers_collection=[],
                 weights_collection=[],
                 use_xavier=False,
                 use_MSRA=False):
    with tf.name_scope(name):
        # Create new weights and biases.
        weights = new_weights(shape=[num_inputs, num_outputs], name=name, use_xavier=use_xavier, use_MSRA=use_MSRA)
        biases = new_biases(length=num_outputs, value=bias_value, name=name)
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

def save_params(sess, file_name, global_step, saver, save_dir='checkpoints/'):
    '''Save the model parameters to the particular directory'''
    # Check if save_dir exists, create the dir if not.
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, file_name)

    saver.save(sess, save_path=save_path, global_step=global_step)

def load_params(sess, model_name, prog, saver, save_dir='checkpoints/'):
    try:
        f = open(save_dir + model_name + '/checkpoint', 'r').readline()
    except FileNotFoundError as identifier:
        raise ValueError("There is no checkpoint found!")
    model_number = prog.search(f).group()
    saver.restore(sess=sess, save_path=save_dir + model_name + '/' + model_number)

def batch_normalization(input_tensor, name, train_status, ema):
    '''Do the batch normalization job and keep the moving average mean and variance'''
    x_shape = input_tensor.get_shape()
    params_shape = x_shape[-1:]

    axis = list(range(len(x_shape) - 1))

    beta = tf.Variable(tf.zeros_initializer()(params_shape), name=name+'beta')
    gamma = tf.Variable(tf.ones_initializer()(params_shape), name=name+'gamma')

    # beta = tf.get_variable(name=name+'beta', 
    #                      shape=params_shape,
    #                      initializer=tf.zeros_initializer)
    # gamma = tf.get_variable(name=name+'gamma',
    #                       shape=params_shape,
    #                       initializer=tf.ones_initializer)

    mean, variance = tf.nn.moments(input_tensor, axis, name=name+'moving_average')
    update_moving_mean = ema.apply([mean])
    update_moving_variance = ema.apply([variance])

    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_mean)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_variance)

    mean, variance = tf.cond(train_status, lambda: (mean, variance), lambda: (ema.average(mean), ema.average(variance)))
    tf.add_to_collection(tf.GraphKeys.BIASES, mean)
    tf.add_to_collection(tf.GraphKeys.BIASES, variance)
    x = tf.nn.batch_normalization(input_tensor, mean, variance, beta, gamma, 0.000001, name=name)

    return x

def new_conv_block(input_tensor, input_channel, kernel_size, filters, stage, block, layers_collection, weights_collection, ema, train_status, stride=2):
    '''This function generates a conv layer group for ResNet with conv layer shortcut
    
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

        # Returns
            Output tensor for the block
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    '''
    filters1, filters2, filters3 = filters
    conv_name_base = 'res_' + str(stage) + block + '_branch'
    bn_name_base = 'bn_' + str(stage) + block + '_branch'
    
    # First conv layer
    x, _ = new_conv_layer(input=input_tensor, 
                       num_input_channels=input_channel, 
                       filter_size=1, 
                       stride=stride,
                       num_filters=filters1,
                       use_MSRA=True,
                       layers_collection=layers_collection,
                       weights_collection=weights_collection,
                       name=conv_name_base+'_a',
                       use_relu=False,
                       )
    # x = batch_normalization(x, name=bn_name_base+'_a', train_status=train_status, ema=ema)
    x = tf.layers.batch_normalization(inputs=x, training=train_status, name=bn_name_base+'_a')
    x = tf.nn.relu(x)
    
    # Second conv layer
    x, _ = new_conv_layer(input=x, 
                       num_input_channels=filters1, 
                       filter_size=kernel_size, 
                       stride=1,
                       num_filters=filters2,
                       use_MSRA=True,
                       layers_collection=layers_collection,
                       weights_collection=weights_collection,
                       name=conv_name_base+'_b',
                       use_relu=False,
                       padding_mode='SAME'
                       ) 
    # x = batch_normalization(x, name=bn_name_base+'_b', train_status=train_status, ema=ema)
    x = tf.layers.batch_normalization(inputs=x, training=train_status, name=bn_name_base+'_b')
    x = tf.nn.relu(x)

    # Third conv layer
    x, _ = new_conv_layer(input=x, 
                       num_input_channels=filters2, 
                       filter_size=1, 
                       stride=1,
                       num_filters=filters3,
                       use_MSRA=True,
                       layers_collection=layers_collection,
                       weights_collection=weights_collection,
                       name=conv_name_base+'_c',
                       use_relu=False
                       ) 
    # x = batch_normalization(x, name=bn_name_base+'_c', train_status=train_status, ema=ema)
    x = tf.layers.batch_normalization(inputs=x, training=train_status, name=bn_name_base+'_c')

    # Short connect
    shortcut, _ = new_conv_layer(input=input_tensor, 
                            num_input_channels=input_channel, 
                            filter_size=1, 
                            stride=stride,
                            num_filters=filters3,
                            use_MSRA=True,
                            layers_collection=layers_collection,
                            weights_collection=weights_collection,
                            name=conv_name_base+'shortcut',
                            use_relu=False,
                            ) 
    # shortcut = batch_normalization(shortcut, name=bn_name_base+'shortcut', train_status=train_status, ema=ema)
    shortcut = tf.layers.batch_normalization(inputs=shortcut, training=train_status, name=bn_name_base+'_shortcut')
    x = tf.add(x, shortcut)
    x = tf.nn.relu(x, name=conv_name_base+'_output')
    layers_collection.append(x)
    return x

def new_identity_block(input_tensor, input_channel, kernel_size, filters, stage, block, layers_collection, weights_collection, ema, train_status):
    '''This function generates an identity layer group for ResNet with conv layer shortcut
    
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

        # Returns
            Output tensor for the block
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    '''
    filters1, filters2, filters3 = filters
    conv_name_base = 'res_' + str(stage) + block + '_branch'
    bn_name_base = 'bn_' + str(stage) + block + '_branch'
    
    # First conv layer
    x, _ = new_conv_layer(input=input_tensor, 
                       num_input_channels=input_channel, 
                       filter_size=1, 
                       stride=1,
                       num_filters=filters1,
                       use_MSRA=True,
                       layers_collection=layers_collection,
                       weights_collection=weights_collection,
                       name=conv_name_base+'_a',
                       use_relu=False,
                       )
    # x = batch_normalization(x, name=bn_name_base+'_a', train_status=train_status, ema=ema)
    x = tf.layers.batch_normalization(inputs=x, training=train_status, name=bn_name_base+'_a')
    x = tf.nn.relu(x)
    
    # Second conv layer
    x, _ = new_conv_layer(input=x, 
                       num_input_channels=filters1, 
                       filter_size=kernel_size, 
                       stride=1,
                       num_filters=filters2,
                       use_MSRA=True,
                       layers_collection=layers_collection,
                       weights_collection=weights_collection,
                       name=conv_name_base+'_b',
                       use_relu=False,
                       padding_mode='SAME'
                       ) 
    # x = batch_normalization(x, name=bn_name_base+'_b', train_status=train_status, ema=ema)
    x = tf.layers.batch_normalization(inputs=x, training=train_status, name=bn_name_base+'_b')
    x = tf.nn.relu(x)

    # Third conv layer
    x, _ = new_conv_layer(input=x, 
                       num_input_channels=filters2, 
                       filter_size=1, 
                       stride=1,
                       num_filters=filters3,
                       use_MSRA=True,
                       layers_collection=layers_collection,
                       weights_collection=weights_collection,
                       name=conv_name_base+'_c',
                       use_relu=False
                       ) 
    # x = batch_normalization(x, name=bn_name_base+'_c', train_status=train_status, ema=ema)
    x = tf.layers.batch_normalization(inputs=x, training=train_status, name=bn_name_base+'_c')

    x = tf.add(x, input_tensor)
    x = tf.nn.relu(x, name=conv_name_base+'_output')
    layers_collection.append(x)
    return x

def pad_sequences(train_sequences, test_sequences):
    input_sequence = train_sequences + test_sequences
    num_tokens = [len(tokens) for tokens in input_sequence]
    num_tokens = np.array(num_tokens)
    mean_tokens = np.mean(num_tokens)
    max_tokens = int(mean_tokens + 2 * np.std(num_tokens))

    x_train_pad = tf.keras.preprocessing.sequence.pad_sequences(train_sequences, maxlen=max_tokens)
    x_test_pad = tf.keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen=max_tokens)


    print("Maximum token number is: {}, {:.2f}% of the input data is maintained".format(max_tokens, np.sum((num_tokens < max_tokens) / len(num_tokens))))
    return x_train_pad, x_test_pad, max_tokens