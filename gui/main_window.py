import tkinter as tk
from tkinter import ttk

class SearchForm(tk.Tk):
    def __init__(self, dataset_list, evaluation_model_list):
        super().__init__()

        self.geometry("400x300")
        self.title("Search Form")

        # Перше випадаюче меню "Датасет"
        self.dataset_label = tk.Label(self, text="Датасет")
        self.dataset_label.pack(pady=5)
        self.dataset_var = tk.StringVar()
        self.dataset_menu = ttk.Combobox(self, textvariable=self.dataset_var, state="readonly", width=30)
        self.dataset_menu['values'] = dataset_list
        self.dataset_menu.pack(pady=5)
        self.dataset_menu.bind("<<ComboboxSelected>>", self.check_selection)

        # Друге випадаюче меню "Спосіб класифікації"
        self.classification_label = tk.Label(self, text="Спосіб класифікації")
        self.classification_label.pack(pady=5)
        self.classification_var = tk.StringVar()
        self.classification_menu = ttk.Combobox(self, textvariable=self.classification_var, state="readonly", width=30)
        self.classification_menu['values'] = evaluation_model_list
        self.classification_menu.pack(pady=5)
        self.classification_menu.bind("<<ComboboxSelected>>", self.check_selection)

        # Кнопка "Пошук"
        self.search_button = tk.Button(self, text="Пошук", state=tk.DISABLED, command=self.search)
        self.search_button.pack(pady=20)

    def check_selection(self, event=None):
        if self.dataset_var.get() and self.classification_var.get():
            self.search_button.config(state=tk.NORMAL)
        else:
            self.search_button.config(state=tk.DISABLED)

    def search(self):
        self.result_data = {
            "dataset": self.dataset_var.get(),
            "evaluation_method": self.classification_var.get()
        }
        self.destroy()  # Закрити форму після натискання кнопки

def get_search_criteria(dataset_list, classification_model_list):
    form = SearchForm(dataset_list, classification_model_list)
    form.mainloop()
    return form.result_data
