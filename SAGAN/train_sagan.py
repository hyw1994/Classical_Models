import tensorflow as tf
from models.models import Generator
from models.models import Discriminator
from losses.sagan_loss import discriminator_loss, generator_loss
from utils.utils import create_flags

# Define model constants.
flags = tf.app.flags
flags = create_flags(flags)

# Guild generator and discriminator model.
generator_net = Generator(dtype=flags.FLAGS.dtype)
generator_optimizer = tf.train.AdamOptimizer(learning_rate=flags.FLAGS.learning_rate_generator, beta1=flags.FLAGS.beta1, beta2=flags.FLAGS.beta2)

discriminator_net = Discriminator(alpha=flags.FLAGS.alpha, dtype=flags.FLAGS.dtype)
discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=flags.FLAGS.learning_rate_discriminator, beta1=flags.FLAGS.beta1, beta2=flags.FLAGS.beta2)

# Print the network structure to show that the model is well built.
generator_net.build(input_shape=(None, 128))
discriminator_net.build(input_shape=(None, 128, 128, 3))

generator_net.summary()
discriminator_net.summary()

