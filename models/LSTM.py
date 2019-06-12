import tensorflow as tf
import numpy as np
from tensorflow.python import keras
from . import utils

class LSTM():
    def __init__(self, num_words, max_tokens):
        super().__init__()
        self.embedding_size = 8
        self.num_words = num_words
        self.max_tokens = max_tokens
        
    def build(self):
        input = keras.layers.Input(shape=(self.max_tokens,), name='input')
        embedding = keras.layers.Embedding(input_dim=self.num_words, 
                                                output_dim=self.embedding_size,
                                                input_length=self.max_tokens,
                                                name='layer_embedding')(input)
        lstm1 = keras.layers.GRU(units=16, return_sequences=True, name='layer_lstm1')(embedding)
        lstm2 = keras.layers.GRU(units=8, return_sequences=True, name='layer_lstm2')(lstm1)
        lstm3 = keras.layers.GRU(units=4, name='layer_lstm3')(lstm2)
        dense = keras.layers.Dense(units=1, activation='sigmoid', name='layer_dense')(lstm3)

        self.model = keras.Model(inputs=input, outputs=dense)
        self.model.compile(loss='binary_crossentropy',
                            optimizer=keras.optimizers.Adam(lr=1e-3),
                            metrics=['accuracy'])
        self.model.summary()

    def train(self, epochs, x_train, y_train, batch_size):
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1,
                                                         patience=5, min_lr=0.0001, min_delta=0.01, verbose=1)
        self.model.fit(x_train, y_train, epochs=epochs, validation_split=0.05, batch_size=batch_size, callbacks=[reduce_lr])