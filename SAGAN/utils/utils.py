def create_flags(flags):
    flags.DEFINE_integer(name='z_size', default=128,
                    help="Input random vector dimension")
    flags.DEFINE_float(name='learning_rate_generator', default=0.0001,
                        help="Learning rate for the generator net")
    flags.DEFINE_float(name='learning_rate_discriminator', default=0.0004,
                        help="Learning rate for the discriminator net")
    flags.DEFINE_integer(name='batch_size', default=64,
                        help="Size of the input batch")
    flags.DEFINE_float(name='alpha', default=0.1,
                        help="Leaky ReLU negative slope")
    flags.DEFINE_float(name='beta1', default=0.0,
                        help="Adam optimizer beta1")
    flags.DEFINE_float(name='beta2', default=0.9,
                        help="Adam optimizer beta2")
    flags.DEFINE_integer(name='total_train_steps', default=600000,
                        help="Total number of training steps")
    flags.DEFINE_string(name='dtype', default="float32",
                        help="Training Float-point precision")
    flags.DEFINE_integer(name='record_summary_after_n_steps', default=500,
                        help="Number of interval steps to recording summaries")
    flags.DEFINE_integer(name='number_of_test_images', default=16,
                        help="Number of test images to generate during evaluation")
    flags.DEFINE_integer(name='model_id', default=24824,
                        help="Load this model if found")
    
    return flags