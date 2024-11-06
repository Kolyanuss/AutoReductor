import numpy as np
import tensorflow as tf
import csv
import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tkinter as tk
from tkinter import filedialog, messagebox

class DataLoader():
    folder_path = ""
    test_size = 0.2
    def __init__(self):
        self.dataset_dict = {"mnist": self.__mnist, "fashion_mnist": self.__fashion_mnist, "cifar10": self.__cifar10, "Import custom dataset": self.__import_custom_dataset }
        
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
    
    def __import_custom_dataset(self):
        x = []
        y = []
        root = tk.Tk()
        root.withdraw()
        self.folder_path = filedialog.askdirectory(title="Оберіть папку")
        root.destroy()
        
        if not os.path.exists(self.folder_path):
            print("Error, folder is'n exist")
            exit(1)
        
        for class_name in os.listdir(self.folder_path):
            class_folder_path = os.path.join(self.folder_path, class_name)
            
            if not os.path.isdir(class_folder_path):
                continue
            
            for image_name in os.listdir(class_folder_path):
                image_path = os.path.join(class_folder_path, image_name)
                img = load_img(image_path)
                x.append(img_to_array(img))
                y.append(class_name)  # припускаємо, що назва папки — це клас

        x = np.array(x)
        y = np.array(y)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.test_size, random_state=42, stratify=y)
        return (x_train, y_train, x_test, y_test)
    
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
    
    def save_reducted_data(X, Y, X_reduced, save_path, data_save_name):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        if not data_save_name.endswith('.csv'):
            data_save_name += '.csv'
        
        file_path = os.path.join(save_path, data_save_name)        
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            
            # Записуємо заголовки
            writer.writerow(['Image_id', 'class', 'original_width', 'original_height', 'original_channel', 'reducted_data'])
            
            # Проходимо через всі зображення та їхні редуковані вектори
            for i, (image, img_class, reduced_data) in enumerate(zip(X, Y, X_reduced)):
                
                if len(image.shape) == 2:
                    original_height, original_width = image.shape
                    original_channel = 1
                elif len(image.shape) == 3:
                    original_height, original_width, original_channel = image.shape                
                
                reducted_data_str = ', '.join(map(str, reduced_data))  # Форматуємо вектор редукованих даних в рядок
                
                # Записуємо рядок в файл
                writer.writerow([i + 1, img_class, original_width, original_height, original_channel, reducted_data_str])

        print(f'Data successfully saved to {file_path}')
        messagebox.showinfo("Info", f'Data successfully saved to {file_path}')

