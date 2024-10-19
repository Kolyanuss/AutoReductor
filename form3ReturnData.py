import tkinter as tk
from tkinter import messagebox

entries = []

def train(window):
    values = [entry.get() for entry in entries]
    try:
        values = [int(value) for value in values]
    except ValueError:
        messagebox.showerror("Помилка", "Всі значення повинні бути числовими")
        return
    print("Train function called with values:", values)
    global input_values
    input_values = values
    messagebox.showinfo("Info", "Редукція даних почалась.")
    window.quit()

def create_form(num_el):
    global entries
    entries = []
    global input_values
    input_values = []

    window = tk.Tk()
    window.title("Форма")

    # Напис вгорі вікна
    label = tk.Label(window, text="Підібрані параметри для алгоритмів редукції:", font=("Arial", 16))
    label.pack(pady=10)

    input_frame = tk.Frame(window)
    input_frame.pack(pady=10)
    # Поля для вводу
    for _ in range(num_el):
        entry = tk.Entry(input_frame)
        entry.pack(side=tk.LEFT, padx=5)
        entries.append(entry)

    # Frame для кнопок
    button_frame = tk.Frame(window)
    button_frame.pack(side=tk.BOTTOM, pady=10, fill=tk.X)

    # Кнопка Вихід
    exit_button = tk.Button(button_frame, text="Вихід", command=window.quit)
    exit_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

    # Кнопка Тренування
    train_button = tk.Button(button_frame, text="Тренування", command=lambda: train(window))
    train_button.pack(side=tk.RIGHT, padx=5, expand=True, fill=tk.X)

    window.mainloop()

def get_input_values():
    return input_values
