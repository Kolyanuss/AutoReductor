import time
import tensorflow as tf
from sklearn.decomposition import TruncatedSVD
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Conv2D, MaxPooling2D, UpSampling2D


class myReducePipline():
    lat_dim_ae = 32
    lat_dim_svd = 70

    def __init__(self) -> None:
        self.create_ae()
        self.create_svd()

    def create_ae(self):
        # Input shape (assuming MNIST data)
        input_shape = (28, 28, 1)
        # Encoder
        inputs = Input(shape=input_shape)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(
            inputs)  # Output: 28x28 (avoid information loss)
        x = MaxPooling2D((2, 2), padding='valid')(x)  # Output: 14x14
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(
            x)  # Output: 14x14 (avoid information loss)
        x = MaxPooling2D((2, 2), padding='valid')(x)  # Output: 7x7
        x = Flatten()(x)
        encoded = Dense(self.lat_dim_ae)(x)

        # Decoder (adjusted based on encoder output)
        # Match the number of channels in the next Conv2D layer
        x = Dense(7 * 7 * 64)(encoded)
        x = Reshape((7, 7, 64))(x)
        x = Conv2D(64, (3, 3), activation='relu',
                   padding='same')(x)  # Output: 7x7
        x = UpSampling2D((2, 2))(x)  # Output: 14x14
        x = Conv2D(32, (3, 3), activation='relu',
                   padding='same')(x)  # Output: 14x14
        # Output: 28x28 (matches input height and width)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid',
                         padding='same')(x)  # Output: 28x28x1

        # Create the autoencoder model
        self.autoencoder = tf.keras.Model(inputs, decoded)

        # Compile the model
        self.autoencoder.compile(optimizer='adam', loss='mse')

    def create_svd(self):
        self.svd = TruncatedSVD(n_components=self.lat_dim_svd)

    def fit_ae_model(self, x_train, x_test):
        '''
        x_train and x_test shoud be original dimmension:
        (n, 28, 28) \n
        Return dimm reshaped to 784 (28*28)
        '''
        x_train = x_train.reshape((-1, 28, 28, 1))
        x_test = x_test.reshape((-1, 28, 28, 1))
        # -1 for batch size (infer from data)

        self.autoencoder.fit(x_train, x_train, epochs=10)
        x_train_ae = self.autoencoder.predict(x_train)
        x_test_ae = self.autoencoder.predict(x_test)
        x_train_ae = x_train_ae.reshape((len(x_train_ae), -1))
        x_test_ae = x_test_ae.reshape((len(x_test_ae), -1))
        return x_train_ae, x_test_ae

    def fit_svd_model(self, x_train, x_test):
        self.svd.fit(x_train)
        x_train_svd = self.svd.transform(x_train)
        x_test_svd = self.svd.transform(x_test)
        return x_train_svd, x_test_svd

    def get_svd_inverse_transform(self, x):
        return self.svd.inverse_transform(x)

    def run_pipeline(self, x_train, x_test):
        # AE
        start_time = time.time()
        x_train, x_test = self.fit_ae_model(x_train, x_test)
        AE_train_time = time.time() - start_time

        # SVD
        start_time = time.time()
        x_train, x_test = self.fit_svd_model(x_train, x_test)
        svd_train_time = time.time() - start_time

        print("AE create + train time: ", AE_train_time, " sec")
        print("SVD create + train time:", svd_train_time, " sec")

        return x_train, x_test
