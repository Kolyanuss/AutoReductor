import tkinter as tk
from tkinter import ttk

class PiplineForm(tk.Tk):
    def __init__(self, best_alg):
        super().__init__()

        self.geometry("500x600")
        self.title("Pipline Form")
        
        self.best_alg = best_alg
        
        # Лічильник кількості блоків
        self.block_count_label = tk.Label(self, text="Кількість блоків")
        self.block_count_label.pack(pady=5)
        self.block_count_var = tk.IntVar(value=1)  # За замовчуванням 1
        self.block_count_spinbox = tk.Spinbox(self, from_=1, to=3, width=10, textvariable=self.block_count_var, command=self.update_blocks)
        self.block_count_spinbox.pack(pady=5)

        # Контейнер для блоків
        self.blocks_frame = tk.Frame(self)
        self.blocks_frame.pack(pady=10)
        
        # Масив для збереження блоків
        self.blocks = []
        
        # Додаємо початкові блоки (1 блок за замовчуванням)
        self.update_blocks()

        # Кнопка "Start"
        self.search_button = tk.Button(self, text="Start", state=tk.DISABLED, command=self.start_button, width=20)
        self.search_button.pack(pady=20)
        
        # Обробка подій для активації кнопки "Start"
        self.bind("<KeyRelease>", self.check_selection)
        self.bind("<ButtonRelease-1>", self.check_selection)

    def update_blocks(self):
        """Оновлює кількість блоків на основі вибраного числа."""
        # Видаляємо всі попередні блоки
        for block in self.blocks:
            block['frame'].destroy()  # Знищуємо сам Frame, який містить блок
        self.blocks = []
        
        # Створюємо нові блоки
        for i in range(self.block_count_var.get()):
            block = self.create_block(i + 1)
            self.blocks.append(block)
        
        # self.check_selection()  # Оновлення стану кнопки після зміни блоків

    def create_block(self, block_num):
        """Створює новий блок з випадаючим меню та полями введення."""
        block_frame = tk.Frame(self.blocks_frame)
        block_frame.pack(pady=10)

        # випадаюче меню 
        alg_label = tk.Label(block_frame, text=f"Найкращі алгоритми (Блок {block_num})")
        alg_label.grid(row=0, column=0, columnspan=3, pady=5)
        alg_var = tk.StringVar()
        alg_menu = ttk.Combobox(block_frame, textvariable=alg_var, state="readonly", width=60)
        alg_menu['values'] = self.best_alg
        alg_menu.grid(row=1, column=0, columnspan=3, pady=5)
        alg_menu.bind("<<ComboboxSelected>>", self.check_selection)

        # Поля для введення min_reduction, max_reduction, step_count
        min_reduction_label = tk.Label(block_frame, text="Min Reduction")
        min_reduction_label.grid(row=2, column=0, padx=5)
        min_reduction_spinbox = tk.Spinbox(block_frame, from_=1, to=100, width=10)
        min_reduction_spinbox.grid(row=3, column=0, padx=5)

        max_reduction_label = tk.Label(block_frame, text="Max Reduction")
        max_reduction_label.grid(row=2, column=1, padx=5)
        max_reduction_spinbox = tk.Spinbox(block_frame, from_=1, to=100, width=10)
        max_reduction_spinbox.grid(row=3, column=1, padx=5)

        step_count_label = tk.Label(block_frame, text="Step Count")
        step_count_label.grid(row=2, column=2, padx=5)
        step_count_spinbox = tk.Spinbox(block_frame, from_=1, to=100, width=10)
        step_count_spinbox.grid(row=3, column=2, padx=5)

        # Зберігаємо всі елементи блоку для подальшого використання
        return {
            "alg_var": alg_var,
            "min_reduction_spinbox": min_reduction_spinbox,
            "max_reduction_spinbox": max_reduction_spinbox,
            "step_count_spinbox": step_count_spinbox,
            "frame": block_frame
        }

    def check_selection(self, event=None):
        """Перевіряє всі блоки на коректність введених даних для активації кнопки Start."""
        valid = True
        for block in self.blocks:
            alg = block['alg_var'].get()
            min_reduction = block['min_reduction_spinbox'].get()
            max_reduction = block['max_reduction_spinbox'].get()
            step_count = block['step_count_spinbox'].get()
            
            # Перевірка, чи всі поля заповнені та коректні
            if not (alg and min_reduction.isdigit() and max_reduction.isdigit() and step_count.isdigit() and int(min_reduction) <= int(max_reduction)):
                valid = False
                break
        
        # Активація/деактивація кнопки "Start"
        if valid:
            self.search_button.config(state=tk.NORMAL)
        else:
            self.search_button.config(state=tk.DISABLED)

    def start_button(self):
        """Збирає результати з усіх блоків і закриває форму."""
        self.result_data = []
        for block in self.blocks:
            block_data = {
                "chosen_algorithm": block['alg_var'].get(),
                "reduction_range": (
                    int(block['min_reduction_spinbox'].get()),
                    int(block['max_reduction_spinbox'].get()),
                    int(block['step_count_spinbox'].get())
                )
            }
            self.result_data.append(block_data)
        self.destroy()  # Закрити форму після натискання кнопки

def get_pipline_algorithms_and_ranges(best_alg_list):
    form = PiplineForm(best_alg_list)
    form.mainloop()
    return form.result_data
