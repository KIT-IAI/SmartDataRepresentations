import tensorflow as tf
import tensorflow.keras.layers as layers


class AutoencoderANN(tf.keras.Model):
    """ Autoencoder using only fully connected layers """

    def __init__(self, data_shape, latent_dim, hidden=[]):
        """ Initialise autoencoder model given input shape, latent dimension, and hidden neurons. """
        super().__init__()
        self.latent_dim = latent_dim
        
        encoder_steps = [layers.Flatten()]
        for number_of_neurons in hidden:
            encoder_steps.append(layers.Dense(number_of_neurons, activation='relu'))
        encoder_steps.append(layers.Dense(latent_dim, activation='tanh'))
        self.encoder = tf.keras.Sequential(encoder_steps)

        hidden.reverse()
        decoder_steps = []
        for number_of_neurons in hidden:
            decoder_steps.append(layers.Dense(number_of_neurons, activation='relu'))
        decoder_steps.append(layers.Dense(data_shape[0] * data_shape[1], activation='tanh'))
        decoder_steps.append(layers.Reshape(data_shape))
        self.decoder = tf.keras.Sequential(decoder_steps)

    def call(self, x):
        """ Infer data given by x through the autoencoder network. """
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded

    def encode(self, x):
        """ Encode input data given by x """
        return self.encoder(x)

    def decode(self, x):
        """ Decode latent representation given by x """
        return self.decoder(x)


class AutoencoderCNN(tf.keras.Model):
    """ Autoencoder using convolutional layers """

    def __init__(self, data_shape, latent_dim=8, hidden=[16]):
        """ Initialise autoencoder model given input shape, latent images, and hidden filters. """
        super().__init__()
        self.latent_dim = latent_dim

        encoder_steps = [layers.Input(shape=data_shape)]
        for number_of_filters in hidden:
            encoder_steps.append(
                layers.Conv2D(number_of_filters, (3,3), activation='relu', padding='same', strides=1)
            )
            encoder_steps.append(
                layers.MaxPooling2D(padding='same', strides=2)
            )
        encoder_steps.append(layers.Conv2D(latent_dim, (3,3), activation='tanh', padding='same', strides=1))
        self.encoder = tf.keras.Sequential(encoder_steps)

        hidden.reverse()
        decoder_steps = []
        for number_of_filters in hidden:
            decoder_steps.append(
                layers.Conv2DTranspose(number_of_filters, kernel_size=3, strides=2, activation='relu', padding='same'),
            )
        decoder_steps.append(layers.Conv2D(1, kernel_size=(3,3), activation='tanh', padding='same'))
        self.decoder = tf.keras.Sequential(decoder_steps)

    def call(self, x):
        """ Infer data given by x through the autoencoder network. """
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded

    def encode(self, x):
        """ Encode input data given by x """
        return self.encoder(x)

    def decode(self, x):
        """ Decode latent representation given by x """
        return self.decoder(x)
