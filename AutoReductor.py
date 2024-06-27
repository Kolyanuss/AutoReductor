from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import math
import os
from Data import DataLoader, DataPreproces
from OptimalParametersSelector import OptimalParametersSelector
from MyReductionModels import Autoencoder
from MyEvaluationModels import get_CNN_model, get_ANN

def plot(df, path_to_save=None, file_name=None):
    fig, ax1 = plt.subplots(figsize=(10, 4))    
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    
    x_layer = df.iloc[:, :-3].apply(lambda x: ",".join(x.astype(str)), axis=1) # combination of params
    ax1.set_xticklabels(x_layer, rotation=90)
    ax1.plot(x_layer, df['mean_test_score'], 'g-')
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
    evaluation_model_list = [MLPClassifier(solver="lbfgs"), SVC(kernel='linear')]
    add_noise = False
    
    def __init__(self, min_reduction = 2, max_reduction = 8, step_count = 2) -> None:
        dl = DataLoader(DataLoader.load_cifar10) # + інші датасети + можливість вибирати датасет
        self.data = dl.get_gata()        
        self.original_dimm = self.data[0].shape[1:]
                        
        dataset_input_shape = self.original_dimm if len(self.original_dimm) > 2 else self.original_dimm + (1,)
        
        self.reduction_model_dict = {
            'AE': (Autoencoder(dataset_input_shape), { 'AE__lat_dim_ae': list(range(50*3,160*3,300*3)) } ),
            # 'SVD': (TruncatedSVD(), {'SVD__n_components': list(range(10,160,40))} )
            'SVD': (TruncatedSVD(), {'SVD__n_components': list(self.create_reduction_range(min_reduction, max_reduction, step_count))} )
            }
        # self.evaluation_model_list = [get_ANN((math.prod(dataset_input_shape),))]
        self.evaluation_model_list = [self.evaluation_model_list[0]]
    
    def create_reduction_range(self, min_reduction, max_reduction, step_count):
        original_dimm_flaten = math.prod(self.original_dimm[:2])
        
        reduction_min = round(math.floor(original_dimm_flaten / max_reduction), -1)
        if reduction_min < 10:
            reduction_min = 10
            
        reduction_max = round(math.floor(original_dimm_flaten / min_reduction), -1)
        if reduction_max > original_dimm_flaten:
            reduction_max = original_dimm_flaten
        
        reduction_step = round(math.floor((reduction_max-reduction_min) / step_count), 0)
        
        return range(reduction_min, reduction_max, reduction_step)
        
    
    def start(self):
        self.data = DataPreproces.normalize_x(self.data)
        # self.data = DataPreproces.unwrapper(self.data)
        noised = ""
        if self.add_noise:
            self.data = DataPreproces.add_gaussian_noise(self.data)
            noised = "_noised"
        print(self.data[0].shape)
        
        for eval_modle in self.evaluation_model_list:
            ops = OptimalParametersSelector(self.data, self.reduction_model_dict, eval_modle)
            ops.find_optimal_param()
            result = ops.get_result()
            save_path = os.path.join(self.results_folder, str(eval_modle))
            save_name = "_".join(self.reduction_model_dict.keys()) + noised + ".png"
            plot(result, save_path, save_name)
        
        # best_pipline = myReducePipline(ops.get_best_sequence(), ops.get_best_parametrs()) # todo
        
AutoReductor().start()