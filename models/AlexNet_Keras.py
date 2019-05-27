'''This is the AlexNet model implemented by Keras API'''
import tensorflow as tf

class AlexNet_Keras(tf.keras.Model):
    def __init__(self, NUM_CLASSES):
        super().__init__()
        # Conv1: 96 11*11 filters, stride 4, relu 0, local response normalization, max pooing with z=3, s=2.
        self.conv1 = tf.keras.layers.Conv2D(filters=96, 
                                            kernel_size=11, 
                                            strides=4, 
                                            activation=tf.keras.layers.ReLU, 
                                            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=0.01))
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))

        # Conv2: pad 2 256 5*5 filters, stride 1, relu 1, local response normalization, max pooing with z=3, s=2.
        self.pad2 = tf.keras.layers.ZeroPadding2D(padding=(2,2))
        self.conv2 = tf.keras.layers.Conv2D(filters=256, 
                                            kernel_size=5, 
                                            strides=1, 
                                            activation=tf.keras.layers.ReLU, 
                                            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=0.01),
                                            bias_initializer=tf.keras.initializers.Ones)
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))

        # Conv3: pad 1 384 3*3 filters, stride 1, relu 0.
        self.pad3 = tf.keras.layers.ZeroPadding2D(padding=(1,1))
        self.conv3 = tf.keras.layers.Conv2D(filters=384,
                                            kernel_size=3,
                                            strides=1,
                                            activation=tf.keras.layers.ReLU, 
                                            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=0.01))
        
        # Conv4: pad1 384 3*3 filters, stride 1,relu 0.
        self.pad4 = tf.keras.layers.ZeroPadding2D(padding=(1,1))
        self.conv4 = tf.keras.layers.Conv2D(filters=384,
                                            kernel_size=3,
                                            strides=1,
                                            activation=tf.keras.layers.ReLU, 
                                            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=0.01))
        
        # Conv5: pad1 384 3*3 filters, stride 1,relu 1, max pooing with z=3, s=2.
        self.pad5 = tf.keras.layers.ZeroPadding2D(padding=(1,1))
        self.conv5 = tf.keras.layers.Conv2D(filters=384,
                                            kernel_size=3,
                                            strides=1,
                                            activation=tf.keras.layers.ReLU, 
                                            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=0.01),
                                            bias_initializer=tf.keras.initializers.Ones)
        self.pool5 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))

        # Flatten
        self.flatten = tf.keras.layers.Flatten()

        # FC6: 4096 neurons, relu 1.
        self.fc6 = tf.keras.layers.Dense(units=4096,
                                         activation=tf.keras.layers.ReLU,
                                         kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=0.01),
                                         bias_initializer=tf.keras.initializers.Ones)
        
        # FC7: 4096 neurons, relu 1.
        self.fc7 = tf.keras.layers.Dense(units=4096,
                                         activation=tf.keras.layers.ReLU,
                                         kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=0.01),
                                         bias_initializer=tf.keras.initializers.Ones)

        # FC8: output sized neurons, relu 1.
        self.fc8 = tf.keras.layers.Dense(units=NUM_CLASSES,
                                         activation=tf.keras.layers.ReLU,
                                         kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=0.01),
                                         bias_initializer=tf.keras.initializers.Ones)

        # Loss function and optimizer.
        loss_object = tf.keras.losses.CategoricalCrossentropy()
        optimizer = tf.keras.optimizers.Adam()

        # Define loss and accuracy for printing over batches.
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = tf.nn.local_response_normalization(input=self.conv1, bias=2, alpha=0.0001, beta=0.75)
        x = self.pool1(x)
        x = self.pad2(x)
        x = self.conv2(x)
        x = tf.nn.local_response_normalization(input=self.conv2, bias=2, alpha=0.0001, beta=0.75)
        x = self.pool2(x)
        x = self.pad3(x)
        x = self.conv3(x)
        x = self.pad4(x)
        x = self.conv4(x)
        x = self.pad5(x)
        x = self.conv5(x)
        x = self.pool5(x)
        x = self.flatten(x)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)

    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            predictions = self(images)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    def test_step(self, images, labels):
        predictions = self(images)
        t_loss = self.loss_object(labels, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)