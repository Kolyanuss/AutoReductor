from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPClassifier
from Data import DataLoader, DataPreproces
from OptimalParametersSelector import OptimalParametersSelector
import matplotlib.pyplot as plt
import pandas as pd
import math
import os
from MyReductionModels import Autoencoder

def plot(df, path_to_save=None, file_name=None):
    fig, ax1 = plt.subplots(figsize=(10, 4))    
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    
    x_layer = df.iloc[:, :-3].apply(lambda x: ",".join(x.astype(str)), axis=1) # combination of params
    ax1.plot(x_layer, df['mean_test_score'], 'g-')
    ax1.set_xticklabels(x_layer, rotation=90)
    ax2.plot(x_layer, df['mean_fit_time'], 'b-')
    ax3.plot(x_layer, df['mean_score_time'], 'r-')

    ax1.set_xlabel(' + '.join(df.iloc[:, :-3].columns))
    ax1.set_ylabel('Eval_model_accuracy', color='g')
    ax2.set_ylabel('fit_time', color='b')
    ax3.set_ylabel('score_time', color='r')

    plt.tight_layout()
    
    if path_to_save and file_name:
        save_fig(path_to_save,file_name)
    else:
        print("File name is not specified")

    plt.show()
    
def save_fig(path_to_save,file_name):
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    plt.savefig(os.path.join(path_to_save,file_name))

class AutoReductor():
    results_folder = "results"
    evaluation_model_list = [MLPClassifier(solver="lbfgs")]
    
    def __init__(self, min_reduction = 5, max_reduction = 20, step_count = 10) -> None:
        dl = DataLoader(DataLoader.load_mnist) # + інші датасети + можливість вибирати датасет
        self.data = dl.get_gata()
        
        original_dimm = math.prod(self.data[0].shape[1:3])
        
        svd_min = round(math.floor(original_dimm / max_reduction), -1)
        if svd_min < 10:
            svd_min = 10
            
        svd_max = round(math.floor(original_dimm / min_reduction), -1)
        if svd_max > original_dimm:
            svd_max = original_dimm
        
        svd_step = round(math.floor((svd_max-svd_min) / step_count), 0)
        
        self.reduction_model_dict = {
            'AE': (Autoencoder(), { 'AE__lat_dim_ae': list(range(10,110,40)) } ),
            # 'SVD': (TruncatedSVD(), {'SVD__n_components': list(range(10,160,40))} )
            'SVD': (TruncatedSVD(), {'SVD__n_components': list(range(svd_min, svd_max, svd_step))} )
            }
    
    def start(self):
        
        
        self.data = DataPreproces.normalize_x(self.data)
        # self.data = DataPreproces.unwrapper(self.data)
        print(self.data[0].shape)
        
        for eval_modle in self.evaluation_model_list:
            ops = OptimalParametersSelector(self.data, self.reduction_model_dict, eval_modle)
            ops.find_optimal_param()
            result = ops.get_result()
            save_path = os.path.join(self.results_folder, str(eval_modle))
            save_name = "_".join(self.reduction_model_dict.keys()) + ".png"
            plot(result, save_path, save_name)
        
        # best_pipline = myReducePipline(ops.get_best_sequence(), ops.get_best_parametrs()) # todo
        
AutoReductor().start()