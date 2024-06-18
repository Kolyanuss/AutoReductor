
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

class OptimalParametersSelector():
    def __init__(self, data, reduction_model_dict, evaluation_model) -> None:
        '''
        Example: 
        steps = [
                ('SVD', TruncatedSVD()),
                ('MLP', MLPClassifier(solver="lbfgs"))]
        pipeline = Pipeline(steps)

        param_grid = {
            'SVD__n_components': list(range(10,160,10)),
        }
        '''
        self.data = data
        steps = []
        self.param_grid = {}
        
        for k,v in reduction_model_dict.items():
            steps.append((k, v[0]))
            self.param_grid.update(v[1])
            
        steps.append(('evaluation', evaluation_model))
        self.pipeline = Pipeline(steps)

        print(steps)
        print(self.param_grid)

    
    def find_optimal_param(self): # todo
        (x_train, y_train, x_test, y_test) = self.data
        
        grid_search = GridSearchCV(self.pipeline, param_grid=self.param_grid, cv=2, scoring='accuracy', return_train_score=True, verbose=2)
        grid_search.fit(x_train, y_train)

        params = ['param_model'+i+'__'+parametr_name[i] for i in range(len(reduction_model_list))]
        self.results = pd.DataFrame(grid_search.cv_results_)[['param_SVD__n_components', 'mean_test_score', 'rank_test_score', 'mean_fit_time', 'mean_score_time']]
