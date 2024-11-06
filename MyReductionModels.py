import math
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

from sklearn.decomposition import PCA, TruncatedSVD, FastICA, NMF
from sklearn.manifold import TSNE

class Autoencoder(BaseEstimator, TransformerMixin):
    def __init__(self, input_shape,  lat_dim_ae=30):
        self.lat_dim_ae = lat_dim_ae
        self.input_shape = input_shape
        self.autoencoder = None
    
    def build_model(self):
        inputs = Input(shape=self.input_shape)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = MaxPooling2D((2, 2), padding='valid')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='valid')(x)
        x = Flatten()(x)
        encoded = Dense(self.lat_dim_ae)(x)

        # Calculate the shape after flattening
        conv_output_shape = (self.input_shape[0] // 4, self.input_shape[1] // 4, 64)
        dense_units = math.prod(conv_output_shape)
        
        x = Dense(dense_units)(encoded)
        x = Reshape(conv_output_shape)(x)
        # x = Dense(7 * 7 * 64)(encoded)
        # x = Reshape((7, 7, 64))(x)        
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(self.input_shape[2], (3, 3), activation='sigmoid', padding='same')(x)

        self.autoencoder = Model(inputs, decoded)
        self.autoencoder.compile(optimizer=Adam(), loss=MeanSquaredError())
    
    def fit(self, X, y=None, **fit_params):
        self.build_model()
        self.autoencoder.fit(X, X, **fit_params)
        return self
    
    def transform(self, X):
        result_x = self.autoencoder.predict(X)
        return result_x.reshape((len(result_x), -1))

class TSNETransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.tsne = TSNE(n_components=n_components)

    def fit(self, X, y=None):
        self.X_transformed = self.tsne.fit_transform(X)
        return self

    def transform(self, X, y=None):
        return self.X_transformed

reduction_model_list = [
    [Autoencoder, TruncatedSVD, PCA, FastICA, TSNETransformer, NMF],
    ["AE__lat_dim_ae", "SVD__n_components", "PCA__n_components", 
     "FastICA__n_components", "TSNE__n_components", "NMF__n_components"]
]

def get_reduction_model_by_name(name):
    for model in reduction_model_list[0]:
        if str(model) == name:
            index = reduction_model_list[0].index(model)
            return model, reduction_model_list[1][index]