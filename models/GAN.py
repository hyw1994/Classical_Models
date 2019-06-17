import tensorflow.python
from keras.layers import *
from keras import Model
from keras.optimizers import RMSprop
from keras import backend as K
# from tensorflow.python.keras.callbacks import TensorBoard
import numpy as np

class GAN():
    def __init__(self, *args, **kwargs):
        self.dropout = 0.4
        self.depth = 64
        return super().__init__(*args, **kwargs)
    
    def build_generator(self, inputs, image_size):
        image_resize = image_size // 4
        # network parameters 
        kernel_size = 5
        layer_filters = [128, 64, 32, 1]

        x = Dense(image_resize * image_resize * layer_filters[0])(inputs)
        x = Reshape((image_resize, image_resize, layer_filters[0]))(x)

        for filters in layer_filters:
            # first two convolution layers use strides = 2
            # the last two use strides = 1
            if filters > layer_filters[-2]:
                strides = 2
            else:
                strides = 1
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2DTranspose(filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding='same')(x)

        x = Activation('sigmoid')(x)
        generator = Model(inputs, x, name='generator')
        return generator


    def build_discriminator(self, inputs):
        kernel_size = 5
        layer_filters = [32, 64, 128, 256]

        x = inputs
        for filters in layer_filters:
            # first 3 convolution layers use strides = 2
            # last one uses strides = 1
            if filters == layer_filters[-1]:
                strides = 1
            else:
                strides = 2
            x = LeakyReLU(alpha=0.2)(x)
            x = Conv2D(filters=filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding='same')(x)

        x = Flatten()(x)
        x = Dense(1)(x)
        x = Activation('sigmoid')(x)
        discriminator = Model(inputs, x, name='discriminator')
        return discriminator

    def build(self):
        # Build the discriminator first.
        input_dis = Input(shape=(28, 28, 1), name='Input_dis')
        self.discriminator = self.build_discriminator(input_dis)
        self.discriminator.compile(optimizer=RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8), loss='binary_crossentropy', metrics=['accuracy'])
        self.discriminator.summary()

        # Build generator here
        input_gen = Input(shape=(100,), name='Input_gen')
        self.generator = self.build_generator(input_gen, 28)
        self.generator.summary()

        # Build the GAN model. 
        self.discriminator.trainable=False
        adversarial = self.discriminator(self.generator(input_gen))
        self.adversarial = Model(inputs=input_gen, outputs=adversarial, name='adversarial')
        self.adversarial.compile(optimizer = RMSprop(lr=0.0004, clipvalue=1.0, decay=3e-8), loss='binary_crossentropy', metrics=['accuracy'])
        self.adversarial.summary()

    def train(self, x_train, x_test):
        batch_size = 128
        images_train = x_train[np.random.randint(0,
                                    x_train.shape[0], size=batch_size), :, :, :]
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
        images_fake = self.generator.predict(noise)
        x = np.concatenate((images_train, images_fake))
        y = np.ones([2*batch_size, 1])
        y[batch_size:, :] = 0
        d_loss = self.discriminator.train_on_batch(x, y)
        print(d_loss)

        y = np.ones([batch_size, 1])
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
        a_loss = self.adversarial.train_on_batch(noise, y)
        print(a_loss)