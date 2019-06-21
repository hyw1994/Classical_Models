import tensorflow as tf
from tensorflow.python.keras import Model
from layers.ResizeImage import ResizeImage
from layers.convolutions import Conv2D, Dense, Conv2DTranspose
from layers.self_attention import SelfAttention

class Generator(tf.keras.Model):
    def __init__(self, dtype):
        super(Generator, self).__init__()
        
        self.fc1 = tf.keras.layers.Dense(units=4 * 4* 512, dtype=dtype, activation="relu")
        self.transp_conv1 = Conv2DTranspose(256, 4, spectral_normalization=True, dtype=dtype, use_bias=False, strides=2,
                                            padding="SAME", activation=None)
        self.bn1 = tf.keras.layers.BatchNormalization(scale=False, dtype=dtype, fused=False, name="bn1")

        self.transp_conv2 = Conv2DTranspose(128, 4, spectral_normalization=True, dtype=dtype, use_bias=False, strides=2,
                                            padding="SAME", activation=None)
        self.bn2 = tf.keras.layers.BatchNormalization(scale=False, dtype=dtype,fused=False, name="bn2")

        self.transp_conv3 = Conv2DTranspose(64, 4, spectral_normalization=True, dtype=dtype, use_bias=False, strides=2,
                                            padding="SAME", activation=None)
        self.bn3 = tf.keras.layers.BatchNormalization(scale=False, dtype=dtype, fused=False, name="bn3")

        # pass the number of filters of the current feature volume
        self.attention = SelfAttention(64, dtype=dtype)

        self.transp_conv4 = Conv2DTranspose(32, 4, spectral_normalization=True, dtype=dtype, use_bias=False, strides=2,
                                            padding="SAME", activation=None)
        self.bn4 = tf.keras.layers.BatchNormalization(scale=False, dtype=dtype, fused=False, name="bn4")

        self.transp_conv5 = Conv2DTranspose(16, 4, spectral_normalization=True, dtype=dtype, use_bias=False, strides=2,
                                            padding="SAME", activation=None)
        self.bn5 = tf.keras.layers.BatchNormalization(scale=False, dtype=dtype, fused=False, name="bn5")

        self.conv = Conv2D(3, 3, strides=1, spectral_normalization=True, dtype=dtype, padding='SAME', activation=None)
        self.out = tf.keras.layers.Activation(activation='tanh')

    def call(self, z, is_training=True):
        net = self.fc1(z)
        net = tf.keras.layers.Reshape((4, 4, 512))(net)

        net = self.transp_conv1(net, training=is_training)
        net = self.bn1(net, training=is_training)
        net = tf.keras.layers.Activation(activation='relu')(net)

        net = self.transp_conv2(net, training=is_training)
        net = self.bn2(net, training=is_training)
        net = tf.keras.layers.Activation(activation='relu')(net)

        net = self.transp_conv3(net, training=is_training)
        net = self.bn3(net, training=is_training)
        net = tf.keras.layers.Activation(activation='relu')(net)

        net = self.attention(net)

        net = self.transp_conv4(net, training=is_training)
        net = self.bn4(net, training=is_training)
        net = tf.keras.layers.Activation(activation='relu')(net)

        net = self.transp_conv5(net, training=is_training)
        net = self.bn5(net, training=is_training)
        net = tf.keras.layers.Activation(activation='relu')(net)

        net = self.conv(net, training=is_training)
        output = self.out(net)

        return output

class Discriminator(tf.keras.Model):
    def __init__(self, alpha, dtype):
        super(Discriminator, self).__init__()
        self.alpha = alpha
        # -------- Block 1
        self.conv1 = Conv2D(32, 4, spectral_normalization=True, dtype=dtype, strides=2, padding='SAME', activation=None)
        self.conv2 = Conv2D(64, 4, spectral_normalization=True, strides=2, dtype=dtype, padding='SAME', activation=None)

        # pass the number of filters of the current feature volume
        self.attention = SelfAttention(64, dtype=dtype)

        self.conv3 = Conv2D(128, 4, spectral_normalization=True, dtype=dtype, strides=2, padding='SAME', activation=None)
        self.conv4 = Conv2D(256, 4, spectral_normalization=True, dtype=dtype, strides=2, padding='SAME', activation=None)
        self.conv5 = Conv2D(512, 4, spectral_normalization=True, dtype=dtype, strides=2, padding='SAME', activation=None)

        self.flat = tf.keras.layers.Flatten()
        self.fc1 = Dense(units=1, spectral_normalization=True, dtype=dtype, activation=None, name="logits")

    def call(self, inputs, is_training=True):
        net = self.conv1(inputs, training=is_training)
        net = tf.keras.layers.LeakyReLU(alpha=self.alpha)(net)

        net = self.conv2(net, training=is_training)
        net = tf.keras.layers.LeakyReLU(alpha=self.alpha)(net)

        net = self.attention(net)

        net = self.conv3(net, training=is_training)
        net = tf.keras.layers.LeakyReLU(alpha=self.alpha)(net)

        net = self.conv4(net, training=is_training)
        net = tf.keras.layers.LeakyReLU(alpha=self.alpha)(net)

        net = self.conv5(net, training=is_training)
        net = tf.keras.layers.LeakyReLU(alpha=self.alpha)(net)

        net = self.flat(net)
        logits = self.fc1(net)

        return logits