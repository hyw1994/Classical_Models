import tensorflow as tf
import pathlib
import scipy.io as scio

def create_flags(flags):
    flags.DEFINE_integer(name='z_size', default=128,
                    help="Input random vector dimension")
    flags.DEFINE_float(name='learning_rate_generator', default=0.0001,
                        help="Learning rate for the generator net")
    flags.DEFINE_float(name='learning_rate_discriminator', default=0.0004,
                        help="Learning rate for the discriminator net")
    flags.DEFINE_integer(name='batch_size', default=64,
                        help="Size of the input batch")
    flags.DEFINE_float(name='alpha', default=0.1,
                        help="Leaky ReLU negative slope")
    flags.DEFINE_float(name='beta1', default=0.0,
                        help="Adam optimizer beta1")
    flags.DEFINE_float(name='beta2', default=0.9,
                        help="Adam optimizer beta2")
    flags.DEFINE_integer(name='total_train_steps', default=600000,
                        help="Total number of training steps")
    flags.DEFINE_string(name='dtype', default="float32",
                        help="Training Float-point precision")
    flags.DEFINE_integer(name='record_summary_after_n_steps', default=500,
                        help="Number of interval steps to recording summaries")
    flags.DEFINE_integer(name='number_of_test_images', default=16,
                        help="Number of test images to generate during evaluation")
    flags.DEFINE_integer(name='model_id', default=1001,
                        help="Load this model if found")
    
    return flags

def download_images():
    image_root = tf.keras.utils.get_file('jpg', 'http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz', untar=True)
    image_root = pathlib.Path(image_root)

    print('*' * 32)
    print("The image set has been downloaded in the path: " + str(image_root))
    print('*' * 32)
    
    return image_root

def load_data(image_root):
    all_image_paths = load_labels_and_image_path(image_root)
    train_path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    train_ds = train_path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return train_ds

def load_and_preprocess_image(path):
    raw_image = tf.read_file(path)
    image = tf.image.decode_jpeg(raw_image, channels=3)

    return image

def load_labels_and_image_path(image_root):
    all_image_paths = list(image_root.glob('*.jpg'))
    all_image_paths = [str(path) for path in all_image_paths]

    return all_image_paths

def prepare_train_ds(train_ds, BATCH_SIZE, INPUT_SIZE, image_size=227):
    """We shuffle and batch the training samples to make the training process work better."""
    # Resize the input image size.
    train_ds = train_ds.map(lambda image : pre_process_ds_images(image, size=image_size)) 

    # We prepare the shuffle buffer to be the same size as the whole size of the input sample to make it shuffle globally.
    train_ds = train_ds.shuffle(buffer_size=INPUT_SIZE)
    train_ds = train_ds.repeat()
    train_ds = train_ds.batch(BATCH_SIZE)
    # Prefetch the data to accerlate the training process. 
    train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_ds

def pre_process_ds_images(image, size=227):
    """ reshape the image to [227, 227] to fit the network."""
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_images(images=image, size=[size, size])
    image = image/255.0

    return image

def save_model(root, checkpoint_prefix):
    root.save(file_prefix=checkpoint_prefix)