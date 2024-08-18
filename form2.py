import tkinter as tk
from tkinter import ttk

class PiplineForm(tk.Tk):
    def __init__(self, best_alg):
        super().__init__()

        self.geometry("400x400")
        self.title("Pipline Form")

        # Перше випадаюче меню 
        self.alg_label = tk.Label(self, text="Найкращі алгоритми")
        self.alg_label.pack(pady=5)
        self.alg_var = tk.StringVar()
        self.alg_menu = ttk.Combobox(self, textvariable=self.alg_var, state="readonly", width=60)
        self.alg_menu['values'] = best_alg
        self.alg_menu.pack(pady=5)
        
        # Створення контейнера для трьох полів
        self.reduction_frame = tk.Frame(self)
        self.reduction_frame.pack(pady=10)
        
        # Поля для введення min_reduction, max_reduction, step_count
        self.min_reduction_label = tk.Label(self.reduction_frame, text="Min Reduction")
        self.min_reduction_label.grid(row=0, column=0, padx=5)
        self.min_reduction_spinbox = tk.Spinbox(self.reduction_frame, from_=1, to=100, width=10)
        self.min_reduction_spinbox.grid(row=1, column=0, padx=5)

        self.max_reduction_label = tk.Label(self.reduction_frame, text="Max Reduction")
        self.max_reduction_label.grid(row=0, column=1, padx=5)
        self.max_reduction_spinbox = tk.Spinbox(self.reduction_frame, from_=1, to=100, width=10)
        self.max_reduction_spinbox.grid(row=1, column=1, padx=5)

        self.step_count_label = tk.Label(self.reduction_frame, text="Step Count")
        self.step_count_label.grid(row=0, column=2, padx=5)
        self.step_count_spinbox = tk.Spinbox(self.reduction_frame, from_=1, to=100, width=10)
        self.step_count_spinbox.grid(row=1, column=2, padx=5)

        # Кнопка "Start"
        self.search_button = tk.Button(self, text="Start", state=tk.DISABLED, command=self.start_button, width=20)
        self.search_button.pack(pady=20)
        
        self.alg_menu.bind("<<ComboboxSelected>>", self.check_selection)
        self.bind("<KeyRelease>", self.check_selection)
        self.bind("<ButtonRelease-1>", self.check_selection)

    def check_selection(self, event=None):
        if (self.alg_var.get() and (self.min_reduction_spinbox.get() <= self.max_reduction_spinbox.get()) and
            self.min_reduction_spinbox.get().isdigit() and self.max_reduction_spinbox.get().isdigit() and 
            self.step_count_spinbox.get().isdigit()):
            self.search_button.config(state=tk.NORMAL)
        else:
            self.search_button.config(state=tk.DISABLED)

    def start_button(self):
        self.result_data = {
            "chosen_algorithm": self.alg_var.get(),
            "reduction_range": (int(self.min_reduction_spinbox.get()), int(self.max_reduction_spinbox.get()), int(self.step_count_spinbox.get()))
        }
        self.destroy()  # Закрити форму після натискання кнопки

def get_pipline_algorithms_and_ranges(best_alg_list):
    form = PiplineForm(best_alg_list)
    form.mainloop()
    return form.result_data
