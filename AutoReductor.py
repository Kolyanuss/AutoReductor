from Data import DataLoader, DataPreproces
from OptimalParametersSelector import OptimalParametersSelector

class AutoReductor():
    def __init__(self) -> None:
        pass
    
    def start(self):
        dl = DataLoader(DataLoader.load_mnist) # + інші датасети + можливість вибирати датасет
        data = dl.get_gata()
        data = DataPreproces.normalize_x(data)
        data = DataPreproces.unwrapper(data)
        print(data[0].shape)
        
        # ops = OptimalParametersSelector(data, reduction_model_list, evaluation_model_list)
        # ops.find_optimal_param()
        
        # best_pipline = ops.get_best_pipline()
        # or
        # best_pipline = myReducePipline(ops.get_best_sequence(), ops.get_best_parametrs())
        
AutoReductor().start()