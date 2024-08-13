import tkinter as tk
from tkinter import ttk

class PiplineForm(tk.Tk):
    def __init__(self, best_alg_list):
        super().__init__()

        self.geometry("400x300")
        self.title("Pipline Form")

        # Перше випадаюче меню 
        self.alg_label = tk.Label(self, text="Найкращі алгоритми")
        self.alg_label.pack(pady=5)
        self.alg_var = tk.StringVar()
        self.alg_menu = ttk.Combobox(self, textvariable=self.alg_var, width=30)
        self.alg_menu['values'] = best_alg_list
        self.alg_menu.pack(pady=5)
        self.alg_menu.bind("<<ComboboxSelected>>", self.check_selection)

        # Кнопка "Start"
        self.search_button = tk.Button(self, text="Start", state=tk.DISABLED, command=self.start_button)
        self.search_button.pack(pady=20)

    def check_selection(self, event=None):
        if self.alg_var.get():
            self.search_button.config(state=tk.NORMAL)
        else:
            self.search_button.config(state=tk.DISABLED)

    def start_button(self):
        self.result_data = {
            "chosen_algorithm": self.alg_var.get(),
            "reduction_range": "none"
        }
        self.destroy()  # Закрити форму після натискання кнопки

def get_pipline_algorithms_and_ranges(best_alg_list):
    form = PiplineForm(best_alg_list)
    form.mainloop()
    return form.result_data
