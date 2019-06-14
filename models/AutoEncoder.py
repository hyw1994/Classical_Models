import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.callbacks import TensorBoard

class AutoEncoder():
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)
    
    def build(self):
        input = keras.layers.Input(shape=(28, 28, 1), name='Input')
        encoder = keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input)
        encoder = keras.layers.MaxPooling2D((2, 2), padding='same')(encoder)
        encoder = keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoder)
        encoder = keras.layers.MaxPooling2D((2, 2), padding='same')(encoder)
        encoder = keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoder)
        encoder = keras.layers.MaxPooling2D((2, 2), padding='same')(encoder)
        
        decoder = keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoder)
        decoder = keras.layers.UpSampling2D((2, 2))(decoder)
        decoder = keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(decoder)
        decoder = keras.layers.UpSampling2D((2, 2))(decoder)
        decoder = keras.layers.Conv2D(16, (3, 3), activation='relu')(decoder)
        decoder = keras.layers.UpSampling2D((2, 2))(decoder)
        decoder = keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(decoder)

        self.autoencoder = keras.Model(input, decoder)
        self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        self.autoencoder.summary()

    def train(self, x_train, x_test):
        self.autoencoder.fit(x_train, x_train,
                            epochs=50,
                            batch_size=128,
                            shuffle=True,
                            validation_data=(x_test, x_test),
                            callbacks=[TensorBoard(log_dir='tmp/autoencoder')])