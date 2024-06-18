from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPClassifier
from Data import DataLoader, DataPreproces
from OptimalParametersSelector import OptimalParametersSelector

class AutoReductor():
    # min_reduction, max_reduction
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
        
        ops = OptimalParametersSelector(data, self.reduction_model_dict, self.evaluation_model_list[0])
        # ops.find_optimal_param()
        
        # best_pipline = myReducePipline(ops.get_best_sequence(), ops.get_best_parametrs())
        
AutoReductor().start()