import models.AlexNet
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import random
from models.AlexNet import AlexNet
from models import utils

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  print("GPU device not found, use cpu instead!")
  # raise SystemError('GPU device not found')
else: print('Found GPU at: {}'.format(device_name))

# Step 1: Load dataset from 102 category flower dataset 
with tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=False)) as sess:
    # Use ImageNet 2012, which has 1000 categories.
    # imageNet_train = tfds.load(name="imagenet2012", split=tfds.Split.TRAIN.subsplit(tfds.percent[: 10]), as_supervised=True)
    
    # Use cifar100, which has 100 categories with size 32*32
    # Preproceess the images and set the hyperparameters
    cifar100_train, cifar100_info = tfds.load(name="cifar100", split=tfds.Split.TRAIN, as_supervised=True, with_info=True)
    
    BATCH_SIZE = 128
    EPOCH = 2
    INPUT_SIZE=cifar100_info.splits["train"].num_examples
    iter_number = (int)(INPUT_SIZE / BATCH_SIZE) + 1
    train_ds, iterator, ds_initializer = utils.prepare_train_ds(cifar100_train, BATCH_SIZE, INPUT_SIZE)


    # Use third party images, this code is no longer fit to this model!
    # image_root, label_root = utils.download_images()
    # train_ds, cv_ds, test_ds = utils.load_data(image_root, label_root)
    # train_ds, iterator initializer = utils.prepare_train_ds(train_ds, BATCH_SIZE, INPUT_SIZE)

    image_batch, label_batch = iterator.get_next()
    sess.run(ds_initializer)
    # plt.imshow(sess.run(image_batch)[0])
    alexnet = AlexNet(cifar100_info.features['label'].num_classes)
    if(tf.test.is_gpu_available):
      with tf.device('/gpu:0'):
        alexnet.build(image_batch, label_batch)
    else:
      alexnet.build(image_batch, label_batch)
    alexnet.save_graph(sess)
    alexnet.train(sess, EPOCH, iter_number)