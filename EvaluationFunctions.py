import os, sys, time, joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

def print_model_size(model_to_save):
    file_path =  'my_model.joblib'
    joblib.dump(model_to_save, file_path)
    
    file_size_bytes = os.path.getsize(file_path)
    file_size_kb = file_size_bytes / 1024
    print(f"Розмір файлу-моделі на диску: {file_size_kb:.2f} KB")
    if os.path.isfile(file_path):
        os.remove(file_path)
        
def my_score(model, x_test, y_test):
    start_time = time.time()
    y_pred = model.predict(x_test)
    print(f"Час затрачений для передбачення всіх x_test({len(x_test)}) - {round(time.time() - start_time, 4)} сек")

    size_in_bytes = sys.getsizeof(model)
    print(f"Розмір моделі в оперативній пам'яті: {size_in_bytes} байт")
    print_model_size(model)
    
    print("classification report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Матриця помилок')
    plt.xlabel('Передбачений клас')
    plt.ylabel('Справжній клас')
    plt.show()
    
def performance_evaluation(model, x_train, y_train, x_test, y_test):
    start_time = time.time()
    model.fit(x_train, y_train)
    print(f"Час навчання моделі - {round(time.time() - start_time, 2)}")
    my_score(model, x_test, y_test)