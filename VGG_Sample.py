import os
os.environ['TF_CPP_MIN_VLOG_LEVEL']='3'

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import random
import numpy as np
from models.VGG import VGG
from models import utils

# Deetect if GPU is ready to use.
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  print("GPU device not found, use cpu instead!")
  # raise SystemError('GPU device not found')
else: print('Found GPU at: {}'.format(device_name))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = False

# Step 1: Load dataset from 102 category flower dataset 
with tf.Session(config=config) as sess:
    # Use cifar100, which has 100 categories with size 32*32
    # Preproceess the images and set the hyperparameters
    cifar100_train, cifar100_info = tfds.load(name="cifar100", split=tfds.Split.TRAIN, as_supervised=True, with_info=True)
    BATCH_SIZE = 256
    EPOCH = 7
    INPUT_SIZE=cifar100_info.splits["train"].num_examples
    BUFFER_SIZE = 8000
    NUM_CLASSES = cifar100_info.features['label'].num_classes
    iter_number = (int)(INPUT_SIZE / BATCH_SIZE) + 1
    train_ds = utils.prepare_train_ds(cifar100_train, BATCH_SIZE, BUFFER_SIZE, image_size=224)

    # Use third party images with 102 categories flowers.
    # BATCH_SIZE = 128
    # EPOCH = 7
    # INPUT_SIZE=8189
    # BUFFER_SIZE = 8000
    # NUM_CLASSES = 102
    # iter_number = (int)(INPUT_SIZE / BATCH_SIZE) + 1
    # image_root, label_root = utils.download_images()
    # train_ds, cv_ds, test_ds = utils.load_data(image_root, label_root)
    # train_ds = utils.prepare_train_ds(train_ds, BATCH_SIZE, BUFFER_SIZE, image_size=224)

    train_numpy = tfds.as_numpy(train_ds)
    vgg16 = VGG(model_type='VGG16', dataset_name='cifar100', num_classes=NUM_CLASSES)
    vgg16.build()
    vgg16.save_graph(sess)
    vgg16.train(sess, EPOCH, iter_number, train_numpy)