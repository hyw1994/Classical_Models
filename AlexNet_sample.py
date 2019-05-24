import models.AlexNet
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from models.AlexNet import AlexNet
from models import utils

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  print("GPU device not found, use cpu instead!")
  # raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

# Step 1: Load dataset from 102 category flower dataset 
with tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=False)) as sess:
    BATCH_SIZE = 5000
    EPOCH = 2
    INPUT_SIZE=8190

    image_root, label_root = utils.download_images()
    train_ds, cv_ds, test_ds = utils.load_data(image_root, label_root)
    train_ds = utils.prepare_train_ds(train_ds, BATCH_SIZE, INPUT_SIZE)

    iterator = tf.data.Iterator.from_structure(train_ds.output_types, train_ds.output_shapes)
    ds_initializer = iterator.make_initializer(train_ds)
    
    image_batch, label_batch = iterator.get_next()
    sess.run(ds_initializer)
    # plt.imshow(sess.run(image_batch)[0])
    alexnet = AlexNet()
    if(tf.test.is_gpu_available):
      with tf.device('/gpu:0'):
        alexnet.build(image_batch, label_batch)
    else:
      alexnet.build(image_batch, label_batch)
    alexnet.save_graph(sess)
    alexnet.train(sess, EPOCH)