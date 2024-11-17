
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

class OptimalParametersSelector():
    def __init__(self, data, steps, param_grid) -> None:
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
        self.steps = steps
        self.param_grid = param_grid
        self.pipeline = Pipeline(steps)

        print(steps)
        print(self.param_grid)

    
    def find_optimal_param(self): # todo
        (x_train, y_train, x_test, y_test) = self.data
        
        grid_search = GridSearchCV(self.pipeline, param_grid=self.param_grid, cv=2, scoring='accuracy', return_train_score=True, verbose=2)
        grid_search.fit(x_train, y_train)

        params = ['param_' + key for key in self.param_grid.keys() ]
        self.results = pd.DataFrame(grid_search.cv_results_)[[*params, 'mean_test_score', 'mean_fit_time', 'mean_score_time']]
    
    def get_result(self):
        return self.results
    
    def plot_accuracy_matrix(self):
        arr = self.results
        if len(arr) != 5:
            return
        heatmap_data = arr.pivot(index=arr.columns[0], columns=arr.columns[1], values='mean_test_score')
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', fmt='.2f')

        plt.title('Heatmap of Accuracy')
        plt.ylabel(arr.columns[0])
        plt.xlabel(arr.columns[1])

        plt.show()
    
    def get_reducted_data(self, param_list):
        # [
        #     ('scaler', StandardScaler()),
            # ('svd', TruncatedSVD(n_components=30)),
        #     ('pca', PCA(n_components=10))
        # ]
        
        steps = self.steps[:-1]
        for step_id, param in zip(range(len(steps)), param_list):
            steps[step_id] = (steps[step_id][0], type(steps[step_id][1])(param))
        
        pipeline = Pipeline(steps)
        
        dataX = np.concatenate((self.data[0], self.data[2]))
        X_reduced = pipeline.fit_transform(dataX)
        print(X_reduced.shape)
        print(X_reduced[0])
        return X_reduced
