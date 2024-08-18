import os
import math
import matplotlib.pyplot as plt

import MyEvaluationModels, MyReductionModels
from Data import DataLoader, DataPreproces
from OptimalParametersSelector import OptimalParametersSelector
from form import get_search_criteria
from form2 import get_pipline_algorithms_and_ranges


def plot(df, path_to_save=None, file_name=None):
    fig, ax1 = plt.subplots(figsize=(10, 4))    
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    
    x_layer = df.iloc[:, :-3].apply(lambda x: ",".join(x.astype(str)), axis=1) # combination of params
    ax1.plot(x_layer, df['mean_test_score'], 'g-')
    ax2.plot(x_layer, df['mean_fit_time'], 'b-')
    ax3.plot(x_layer, df['mean_score_time'], 'r-')

    ax1.set_xlabel(' + '.join(df.iloc[:, :-3].columns))
    ax1.set_ylabel('Eval_model_accuracy', color='g')
    ax2.set_ylabel('fit_time', color='b')
    ax3.set_ylabel('score_time', color='r')
    
    ax1.set_xticks(range(len(x_layer)))
    ax1.set_xticklabels(x_layer, rotation=90)

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
    add_noise = False
    
    def __init__(self, dataset, evaluation_model, reduction_algorithms, reduction_range):
        self.data = dataset    
        self.original_dimm = self.data[0].shape[1:]
                        
        dataset_input_shape = self.original_dimm if len(self.original_dimm) > 2 else self.original_dimm + (1,)
        algorithm_range = list(self.create_reduction_range(*reduction_range))
        # svd_range = list(self.create_reduction_range(min_reduction_svd, max_reduction_svd, step_count_svd))
        self.reduction_model_dict = {
            str(reduction_algorithms[1]).split("__")[0] : (reduction_algorithms[0](dataset_input_shape), { reduction_algorithms[1]: algorithm_range } ),
            # 'SVD': (TruncatedSVD(), {'SVD__n_components': svd_range } )
            }
        self.evaluation_model = evaluation_model
    
    def create_reduction_range(self, min_reduction, max_reduction, step_count):
        original_dimm_flaten = math.prod(self.original_dimm[:2])
        
        range_min = round(math.floor(original_dimm_flaten / max_reduction), -1)
        if range_min < 10:
            range_min = 10
            
        range_max = round(math.floor(original_dimm_flaten / min_reduction), -1)
        if range_max > original_dimm_flaten:
            range_max = original_dimm_flaten
        
        range_step = round(math.floor((range_max-range_min) / step_count), 0)
        
        return range(range_min, range_max, range_step)
        
    
    def start(self):
        self.data = DataPreproces.normalize_x(self.data)
        if "TruncatedSVD" in str(list(self.reduction_model_dict.values())[0][0]):
            self.data = DataPreproces.unwrapper(self.data)
        noised = ""
        if self.add_noise:
            self.data = DataPreproces.add_gaussian_noise(self.data)
            noised = "_noised"
        print(self.data[0].shape)
        
        ops = OptimalParametersSelector(self.data, self.reduction_model_dict, self.evaluation_model)
        ops.find_optimal_param()
        ops.plot_accuracy_matrix()
        result = ops.get_result()
        save_path = os.path.join(self.results_folder, str(self.evaluation_model))
        save_name = "_".join(self.reduction_model_dict.keys()) + noised + ".png"
        plot(result, save_path, save_name)
        

if __name__ == "__main__":
    # show form1
    selected_data = get_search_criteria(DataLoader().get_datasets_names(), MyEvaluationModels.evaluation_model_list)
    print(selected_data)
    chosen_dataset = DataLoader().get_data_by_name(selected_data["dataset"])
    chosen_evaluation_method = MyEvaluationModels.get_eval_model_by_name(selected_data["evaluation_method"])
    
    # analyse chosen_dataset    
    best_alg_list = MyReductionModels.reduction_model_list[0] # temp - all
    
    # show form2 with pipline creation
    selected_data2 = get_pipline_algorithms_and_ranges(best_alg_list)
    print(selected_data2)    
    chosen_algs = MyReductionModels.get_reduction_model_by_name(selected_data2["chosen_algorithm"])
    reduction_ranges = selected_data2["reduction_range"]
    
    AutoReductor(chosen_dataset, chosen_evaluation_method, chosen_algs, reduction_ranges).start()
