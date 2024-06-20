from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPClassifier
from Data import DataLoader, DataPreproces
from OptimalParametersSelector import OptimalParametersSelector
import matplotlib.pyplot as plt
import pandas as pd
import os

def plot(df, path_to_save=None, file_name=None):
    fig, ax1 = plt.subplots(figsize=(8, 4))    
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))

    ax1.plot(df['param_SVD__n_components'], df['mean_test_score'], 'g-')
    ax2.plot(df['param_SVD__n_components'], df['mean_fit_time'], 'b-')
    ax3.plot(df['param_SVD__n_components'], df['mean_score_time'], 'r-')

    ax1.set_xlabel('param_SVD__n_components')
    ax1.set_ylabel('mean_test_score', color='g')
    ax2.set_ylabel('mean_fit_time', color='b')
    ax3.set_ylabel('mean_score_time', color='r')

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
    #todo: min_reduction, max_reduction
    reduction_model_dict = {
        'SVD': (TruncatedSVD(), {'SVD__n_components': list(range(10,160,10))} )
        }
    evaluation_model_list = [MLPClassifier(solver="lbfgs")]
    
    def __init__(self) -> None:
        pass
    
    def start(self):
        dl = DataLoader(DataLoader.load_mnist) # + інші датасети + можливість вибирати датасет
        data = dl.get_gata()
        data = DataPreproces.normalize_x(data)
        data = DataPreproces.unwrapper(data)
        print(data[0].shape)
        
        for eval_modle in self.evaluation_model_list:
            ops = OptimalParametersSelector(data, self.reduction_model_dict, eval_modle)
            ops.find_optimal_param()
            result = ops.get_result()
            save_path = os.path.join(self.results_folder, str(eval_modle))
            save_name = "_".join(self.reduction_model_dict.keys()) + ".png"
            plot(result, save_path, save_name)
        
        # best_pipline = myReducePipline(ops.get_best_sequence(), ops.get_best_parametrs()) # todo
        
AutoReductor().start()