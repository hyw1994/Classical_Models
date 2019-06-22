import tensorflow as tf
import tensorflow_datasets as tfds
import os
from models.models import Generator
from models.models import Discriminator
from libs.sagan_loss import discriminator_loss, generator_loss
import utils

tf.enable_eager_execution()
assert tf.__version__ == '1.13.1' or tf.__version__ == '1.14.0', "This code is only tested on Tensorflow 1.13.1 and 1.14.0"
# Define model constants.
flags = tf.app.flags
flags = utils.create_flags(flags)

# Use third party images with 102 categories flowers.
BATCH_SIZE = 128
EPOCH = 7
INPUT_SIZE=8189
BUFFER_SIZE = 8000
NUM_CLASSES = 102
iter_number = (int)(INPUT_SIZE / BATCH_SIZE) + 1
image_root = utils.download_images()
train_ds = utils.load_data(image_root)
train_ds = utils.prepare_train_ds(train_ds, BATCH_SIZE, BUFFER_SIZE, image_size=128)

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

# Save model parameters and tensorboard information the adversarial network
basepath = "./mnist/" + str(flags.FLAGS.model_id)
logdir = os.path.join(basepath, "logs")
print("Base folder:", basepath)
tf_board_writer = tf.contrib.summary.create_file_writer(logdir)
tf_board_writer.set_as_default()

global_step = tf.train.get_or_create_global_step()

gen_checkpoint_dir = os.path.join(basepath, "generator")
gen_checkpoint_prefix = os.path.join(gen_checkpoint_dir, "model.ckpt")

gen_root = tf.train.Checkpoint(optimizer=generator_optimizer,
                          model=generator_net)

disc_checkpoint_dir = os.path.join(basepath, "discriminator")
disc_checkpoint_prefix = os.path.join(disc_checkpoint_dir, "model.ckpt")
disc_root = tf.train.Checkpoint(optimizer=discriminator_optimizer,
                           model=discriminator_net)

if os.path.exists(basepath):
    try:
        gen_root.restore(tf.train.latest_checkpoint(gen_checkpoint_dir))
        print("Generator model restored")
    except Exception as ex:
        print("Error loading the Generator model:", ex)
    
    try:
        disc_root.restore(tf.train.latest_checkpoint(disc_checkpoint_dir))
        print("Discriminator model restored")
    except Exception as ex:
        print("Error loading the Discriminator model:", ex)
    print("Current global step:", tf.train.get_or_create_global_step().numpy())

else:
    print("Model folder not found.")

# Dynamicly train the network
fake_input_test = tf.random_normal(shape=(flags.FLAGS.number_of_test_images, flags.FLAGS.z_size), dtype=flags.FLAGS.dtype)

for epoch in range(EPOCH):
    for batch, batch_real_images in enumerate(train_ds):
        fake_input = tf.random_normal(shape=(flags.FLAGS.batch_size, flags.FLAGS.z_size), dtype=flags.FLAGS.dtype)
        with tf.contrib.summary.record_summaries_every_n_global_steps(flags.FLAGS.record_summary_after_n_steps):
            with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
                # run the generator with the random noise batch
                g_model = generator_net(fake_input, is_training=True)

                # run the discriminator with real input images
                d_logits_real = discriminator_net(batch_real_images, is_training=True)

                # run the discriminator with fake input images (images from the generator)
                d_logits_fake = discriminator_net(g_model, is_training=True)

                # compute the generator loss
                gen_loss = generator_loss(d_logits_fake)

                # compute the discriminator loss
                dis_loss = discriminator_loss(d_logits_real, d_logits_fake)
        
        # get all the discriminator variables
        discriminator_variables = discriminator_net.variables
        discriminator_variables.append(discriminator_net.attention.gamma)

        discriminator_grads = d_tape.gradient(dis_loss, discriminator_variables)

        # get all the discriminator variables
        generator_variables = generator_net.variables

        generator_grads = g_tape.gradient(gen_loss, generator_variables)

        discriminator_optimizer.apply_gradients(zip(discriminator_grads, discriminator_variables),
                                                global_step=global_step)

        generator_optimizer.apply_gradients(zip(generator_grads, generator_variables),
                                            global_step=global_step)

        print("Epoch:{}, Batch: {}, Generator_loss:{:.2f}, Discriminator:{:.2f}".format(epoch+1, batch+1, gen_loss, dis_loss))
        
        counter = global_step.numpy()

        if counter % 2000 == 0:
            print("Current step:", counter)
            with tf.contrib.summary.always_record_summaries():
                generated_samples = generator_net(fake_input_test, is_training=False)
                tf.contrib.summary.image('test_generator_image', tf.to_float(generated_samples), max_images=16)

        if counter % 20000 == 0:
            # save and download the mode
            utils.save_model(gen_root, gen_checkpoint_prefix)
            utils.save_model(disc_root, disc_checkpoint_prefix)

        if counter >= flags.FLAGS.total_train_steps:
            utils.save_model(gen_root, gen_checkpoint_prefix)
            utils.save_model(disc_root, disc_checkpoint_prefix)
            break
    
    tf.contrib.summary.scalar('generator_loss', gen_loss)
    tf.contrib.summary.scalar('discriminator_loss', dis_loss)
    tf.contrib.summary.image('generator_image', tf.to_float(g_model), max_images=5)