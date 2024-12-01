from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

evaluation_model_list = (LogisticRegression(), MLPClassifier(solver="lbfgs"), SVC(kernel='linear'), DecisionTreeClassifier(), RandomForestClassifier())

def get_eval_model_by_name(name):
    for model in evaluation_model_list:
        if name == str(model):
            return model
        