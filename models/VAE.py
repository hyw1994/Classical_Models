import keras
from keras.callbacks import TensorBoard

import tensorflow as tf

class VAE():
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
        
        shape = keras.backend.int_shape(encoder)
        middle = keras.layers.Flatten()(encoder)
        middle = keras.layers.Dense(16, activation='relu')(middle)
        z_mean = keras.layers.Dense(2, name='z_mean')(middle)
        z_log_var = keras.layers.Dense(2, name='z_log_var')(middle)
        z = keras.layers.Lambda(self.sampling, name='z')([z_mean, z_log_var])
        self.encoded = keras.Model(input, [z_mean, z_log_var, z], name='encoder')
        self.encoded.summary()
        keras.utils.plot_model(self.encoded, to_file='vae_cnn_encoder.png', show_shapes=True)

        latent_input = keras.layers.Input(shape=(2,), name='z_sampling')
        decoder = keras.layers.Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_input)
        decoder = keras.layers.Reshape((shape[1], shape[2], shape[3]))(decoder)
        decoder = keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(decoder)
        decoder = keras.layers.UpSampling2D((2, 2))(decoder)
        decoder = keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(decoder)
        decoder = keras.layers.UpSampling2D((2, 2))(decoder)
        decoder = keras.layers.Conv2D(16, (3, 3), activation='relu')(decoder)
        decoder = keras.layers.UpSampling2D((2, 2))(decoder)
        decoder = keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(decoder)
        self.decoded = keras.Model(latent_input, decoder, name='decoder')
        self.decoded.summary()
        keras.utils.plot_model(self.decoded, to_file='vae_cnn_decoder.png', show_shapes=True)

        outputs = self.decoded(self.encoded(input)[2])
        self.vae = keras.Model(inputs=input, outputs=outputs, name='vae')
        vae_loss = self.reconstruction_loss(input, outputs, z_mean, z_log_var)

        # reconstruction_loss = keras.losses.mse(keras.backend.flatten(input), keras.backend.flatten(outputs))
        # reconstruction_loss *= 28 * 28
        # kl_loss = 1 + z_log_var - keras.backend.square(z_mean) - keras.backend.exp(z_log_var)
        # kl_loss = keras.backend.sum(kl_loss, axis=-1)
        # kl_loss *= -0.5
        # vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)

        # loss = self.reconstruction_loss(input, outputs, z_mean, z_log_var)
        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer='rmsprop')

        self.vae.summary()
        keras.utils.plot_model(self.vae, to_file='vae_cnn.png', show_shapes=True)

    def sampling(self, args):
        z_mean, z_log_var = args
        batch = keras.backend.shape(z_mean)[0]
        dim = keras.backend.int_shape(z_mean)[1]

        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon

    def reconstruction_loss(self, y_true, y_pred, z_mean, z_log_var):
        reconstruction_loss = keras.losses.mse(keras.backend.flatten(y_true), keras.backend.flatten(y_pred))
        reconstruction_loss *= 28*28
        kl_loss = 1 + z_log_var - keras.backend.square(z_mean) - keras.backend.exp(z_log_var)
        kl_loss = keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)
        return vae_loss

    def train(self, x_train, x_test):
        self.vae.fit(x_train,
                            epochs=50,
                            batch_size=128,
                            shuffle=True,
                            validation_data=(x_test, None),
                            callbacks=[TensorBoard(log_dir='tmp/autoencoder')])