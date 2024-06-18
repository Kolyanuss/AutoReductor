import tensorflow as tf

class DataLoader():
    def __init__(self, dataset) -> None:        
        dataset(self)
    
    def load_mnist(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        self.data = (x_train, y_train, x_test, y_test)
    
    def get_gata(self):
        return self.data
    
class DataPreproces():    
    def normalize_x(data):
        (x_train, y_train, x_test, y_test) = data
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        return (x_train, y_train, x_test, y_test)
    
    def unwrapper(data):
        (x_train, y_train, x_test, y_test) = data
        x_train_flat = x_train.reshape((len(x_train), -1))
        x_test_flat = x_test.reshape((len(x_test), -1))
        return (x_train_flat, y_train, x_test_flat, y_test)