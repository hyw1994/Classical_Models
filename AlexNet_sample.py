import models.AlexNet
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from models.AlexNet import AlexNet
from models import utils

# Step 1: Load dataset from 102 category flower dataset 
with tf.Session() as sess:
    BATCH_SIZE = 128
    EPOCH = 1

    image_root, label_root = utils.download_images()
    train_ds, cv_ds, test_ds = utils.load_data(image_root, label_root)
    
    ds = train_ds.shuffle(buffer_size=8190)
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    iterator = tf.data.Iterator.from_structure(ds.output_types, ds.output_shapes)
    ds_initializer = iterator.make_initializer(ds)
    
    image_batch, label_batch = iterator.get_next()
    sess.run(ds_initializer)
    # plt.imshow(sess.run(image_batch)[0])
    alexnet = AlexNet()
    alexnet.build(image_batch, label_batch)
    alexnet.train(sess, EPOCH)