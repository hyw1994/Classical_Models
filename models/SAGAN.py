from keras.layers import *
from keras.optimizers import Adam
from keras import Model
from layers.SpectralNormalization import DenseSN, ConvSN2D
from layers.ResizeImage import ResizeImage

class SAGAN():
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)

    def make_generator_model(self, noise_input, image_size):
        '''Generate the fake image using noise_input'''
        x = DenseSN(7*7*256, use_bias=False, kernel_initializer='glorot_uniform')(noise_input)
        x = Reshape((7, 7, 256))(x)
        x = ResizeImage(image_size, image_size)(x)
        x = ConvSN2D(3, (1, 1), strides=1, padding='SAME')(x)
        model = Model(noise_input, x)
        return model

    def make_discriminator_model(self):

        return

    def build(self, image_size):
        '''Build the generator and discriminator together.'''
        noise_input = Input(shape=(100,))
        self.generator = self.make_generator_model(noise_input, image_size)
        self.generator.summary()

    def train():
        return
