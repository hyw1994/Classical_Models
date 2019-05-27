from models.AlexNet_Keras import AlexNet_Keras
from models import utils
import tensorflow_datasets as tfds
import tensorflow as tf

tf.enable_eager_execution()
# Use cifar100, which has 100 categories with size 32*32
# Preproceess the images and set the hyperparameters
cifar100_train, cifar100_info = tfds.load(name="cifar100", split=tfds.Split.TRAIN, as_supervised=True, with_info=True)

BATCH_SIZE = 128
EPOCH = 2
# INPUT_SIZE=cifar100_info.splits["train"].num_examples
INPUT_SIZE = 8000
iter_number = (int)(INPUT_SIZE / BATCH_SIZE) + 1
train_ds = utils.prepare_train_ds(cifar100_train, BATCH_SIZE, INPUT_SIZE)

alexnet = AlexNet_Keras(cifar100_info.features['label'].num_classes)

for epoch in range(EPOCH):
    image_batch, label_batch = train_ds.take(BATCH_SIZE)
    alexnet.train_step(image_batch, label_batch)

#   for test_images, test_labels in test_ds:
#     alexnet.test_step(test_images, test_labels)

template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
print (template.format(epoch+1,
                        alexnet.train_loss.result(),
                        alexnet.train_accuracy.result()*100,
                        alexnet.test_loss.result(),
                        alexnet.test_accuracy.result()*100))