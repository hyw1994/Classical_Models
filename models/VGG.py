''' This is the VGG model from the original paper for study purpose.'''
import tensorflow as tf
import csv
import matplotlib.pyplot as plt
import numpy as np
from . import utils

# VGG network architecture.
# [224, 224, 3] INPUT LAYER: Subtracting the mean RGB value for training set. Remember the mean values for evaluation.
# []
# [4096] FC: 4096 neurons, ReLU.
# [4096] FC: 4096 neurons, ReLU.
# [1000] FC: 1000 neurons, Softmax.

class VGG():
    def __init__(self, num_classes, type='VGG16'):
        super().__init__()
        # Specify the network type, VGG16 or VGG19.
        self.model_type = type
        self.num_classes = num_classes
        # Input image.
        self.x_image = tf.placeholder(tf.float32, shape=[None, 227, 227, 3])
        self.y_true = tf.placeholder(tf.int64, shape=[None])
        

