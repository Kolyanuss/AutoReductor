import numpy as np
import tensorflow as tf

class DataLoader():
    def __init__(self):
        self.dataset_dict = {"mnist": self.__mnist, "fashion_mnist": self.__fashion_mnist, "cifar10": self.__cifar10 }
        
    def get_datasets_names(self):
        return list(self.dataset_dict.keys())
    
    def __mnist(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        return (x_train, y_train, x_test, y_test)
        
    def __fashion_mnist(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        return (x_train, y_train, x_test, y_test)
        
    def __cifar10(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        return (x_train, y_train.ravel(), x_test, y_test.ravel())
    
    
    def get_data_by_name(self, name):
        try:
            data_loader_func = self.dataset_dict[name]
            return data_loader_func()
        except Exception as e:
            print("Error in Data module:")
            print(e)
            exit(1)
    
    
class DataPreproces():    
    def normalize_x(data: tuple):
        (x_train, y_train, x_test, y_test) = data
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        return (x_train, y_train, x_test, y_test)
    
    def unwrapper(data: tuple):
        '''
            Return 1d arr for each x item
            Example:
            (60000,10,10,3) -> (60000, 300)
        '''
        (x_train, y_train, x_test, y_test) = data
        x_train_flat = x_train.reshape((len(x_train), -1))
        x_test_flat = x_test.reshape((len(x_test), -1))
        return (x_train_flat, y_train, x_test_flat, y_test)
    
    def add_gaussian_noise(data: tuple, mean=0.0, std=0.1):
        (x_train, y_train, x_test, y_test) = data
        noise = np.random.normal(mean, std, x_train.shape)
        x_train_noise = x_train + noise
        noise = np.random.normal(mean, std, x_test.shape)
        x_test_noise = x_test + noise
        return (x_train_noise, y_train, x_test_noise, y_test)