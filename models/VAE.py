from tensorflow.python import keras
from tensorflow.python.keras.callbacks import TensorBoard

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
        self.encoder = keras.Model(input, [z_mean, z_log_var, z], name='encoder')
        self.encoder.summary()
        keras.utils.plot_model(self.encoder, to_file='vae_cnn_encoder.png', show_shapes=True)

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
        self.decoder = keras.Model(latent_input, decoder, name='decoder')
        self.decoder.summary()
        keras.utils.plot_model(self.decoder, to_file='vae_cnn_decoder.png', show_shapes=True)

        outputs = self.decoder(self.encoder(input)[2])
        self.vae = keras.Model(inputs=input, outputs=outputs, name='vae')
        vae_loss = self.reconstruction_loss(z_mean, z_log_var)
        self.vae.compile(optimizer='rmsprop', loss=vae_loss)
        self.vae.summary()
        keras.utils.plot_model(self.vae, to_file='vae_cnn.png', show_shapes=True)

    def sampling(self, args):
        z_mean, z_log_var = args
        batch = keras.backend.shape(z_mean)[0]
        dim = keras.backend.int_shape(z_mean)[1]

        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon

    def reconstruction_loss(self, z_mean, z_log_var):
        kl_loss = 1 + z_log_var - keras.backend.square(z_mean) - keras.backend.exp(z_log_var)
        kl_loss = keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        def loss(y_true, y_pred):
            reconstruction_loss = keras.losses.mse(keras.backend.flatten(y_true), keras.backend.flatten(y_pred))
            reconstruction_loss *= 28*28
            vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)
            return vae_loss

        return loss

    def train(self, x_train, x_test):
        self.vae.fit(x_train, x_train,
                            epochs=50,
                            batch_size=128,
                            shuffle=True,
                            validation_data=(x_test, x_test),
                            callbacks=[TensorBoard(log_dir='tmp/VAE')])