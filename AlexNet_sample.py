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
    image_label_ds = utils.load_data(image_root, label_root)
    ds = image_label_ds.shuffle(buffer_size=8190)
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)

    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    alexnet = AlexNet()
    alexnet.build()

    for epoch in range(EPOCH):
        for image_batch, label_batch in iter(ds):
            alexnet.train(image_batch, label_batch)
