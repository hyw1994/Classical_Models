from tensorflow.python import keras
from tensorflow.python.keras.datasets import imdb
from models.LSTM import LSTM
from models import utils
import tensorflow as tf
import numpy as np

# Deetect if GPU is ready to use.
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  print("GPU device not found, use cpu instead!")
  # raise SystemError('GPU device not found')
else: print('Found GPU at: {}'.format(device_name))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = False
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

# # save np.load
# np_load_old = np.load
# # modify the default parameters of np.load
# np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
num_words = 20000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

epoch = 5
batch_size = 64

word_to_index = imdb.get_word_index()
word_to_index = {key:(value+3) for key,value in word_to_index.items()}
word_to_index["<PAD>"] = 0
word_to_index["<START>"] = 1
word_to_index["<UNK>"] = 2
index_to_word = {value:key for key,value in word_to_index.items()}

def print_sentence(id_list):
    print(' '.join([index_to_word[id] for id in id_list if id != 0]))

print("Train-set size: ", len(x_train))
print("Test-set size:  ", len(x_test))

x_train, x_test, max_tokens = utils.pad_sequences(train_sequences=x_train, test_sequences=x_test)

lstm = LSTM(num_words, max_tokens)
lstm.build()
lstm.train(epochs=epoch, x_train=x_train, y_train=y_train, batch_size=batch_size)