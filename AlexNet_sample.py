import models.AlexNet
import tensorflow as tf
import pathlib
from models.AlexNet import AlexNet
import matplotlib.pyplot as plt
# Step 1: Load dataset from 102 category flower dataset 
data_root = tf.keras.utils.get_file('jpg', 'http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz', untar=True)
data_root = pathlib.Path(data_root)
print(data_root)

alexnet = AlexNet()
alexnet.build()

