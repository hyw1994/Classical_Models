import keras
from .SpectralNormalization import ConvSN2D
from keras.layers import Flatten
import tensorflow as tf
# tfe = tf.contrib.eager
# from libs.convolutions import Conv2D

def hw_flatten(x):
    # Input shape x: [BATCH, HEIGHT, WIDTH, CHANNELS]
    # flat the feature volume across the width and height dimensions 
    x_shape = keras.backend.shape(x)
    return keras.backend.reshape(x, (x_shape[0], -1, x_shape[-1])) # return [BATCH, W*H, CHANNELS]

class SelfAttention(tf.keras.Model):
    def __init__(self, number_of_filters, dtype=tf.float64):
        super(SelfAttention, self).__init__()
        
        self.f = ConvSN2D(number_of_filters//8, 1, spectral_normalization=True,
                                        strides=1, padding='SAME', name="f_x",
                                        activation=None, dtype=dtype)
        
        self.g = ConvSN2D(number_of_filters//8, 1, spectral_normalization=True,
                                        strides=1, padding='SAME', name="g_x",
                                        activation=None, dtype=dtype)
        
        self.h = ConvSN2D(number_of_filters, 1, spectral_normalization=True,
                                        strides=1, padding='SAME', name="h_x",
                                        activation=None, dtype=dtype)
        
        self.gamma = tf.Variable(0., dtype=dtype, trainable=True, name="gamma")
        self.flatten = tf.keras.layers.Flatten()
        
    def call(self, x):

        f = self.f(x)
        g = self.g(x)
        h = self.h(x)
        
        f_flatten = hw_flatten(f)
        g_flatten = hw_flatten(g)
        h_flatten = hw_flatten(h)
        
        s = tf.matmul(g_flatten, f_flatten, transpose_b=True) # [B,N,C] * [B, N, C] = [B, N, N]

        b = tf.nn.softmax(s, axis=-1)
        o = tf.matmul(b, h_flatten)
        y = self.gamma * tf.reshape(o, tf.shape(x)) + x

        return y