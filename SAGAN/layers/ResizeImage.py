from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras import backend as K

class ResizeImage(Layer):
    def __init__(self, width, height, interpolation='nearest'):
        super().__init__()
        self.width = width
        self.height = height
        self.interpolation = interpolation

    def build(self, input_shape):
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        return K.resize_images(inputs, self.height, self.width, data_format="channels_last", interpolation=self.interpolation)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.height, self.width, input_shape[-1])