# Data Reduction Application

## Мета додатку

Цей додаток створений для допомоги користувачам в експериментах із редукцією даних, використовуючи різні алгоритми зниження розмірності та класифікації. Виконуйте кожен крок, щоб дослідити, як різні алгоритми впливають на зменшені дані.

---

## Інструкція користувачу

### Крок 1: Вибір датасету та моделі класифікації

1. **Запуск додатку**  
   Після запуску на головному екрані (Рисунок 4.1) знайдіть два випадаючі меню:
   - **Датасет:** Виберіть набір даних зі списку.
   - **Метод класифікації:** Оберіть алгоритм для оцінки якості редукції.

    ![Рисунок 4.1 – Головний екран](/media/pic1.png) <!-- Вставте посилання на зображення -->

2. **Підтвердження вибору**  
   Натисніть кнопку **"Пошук"**, щоб перейти до екрана налаштувань редукції. Якщо обрано завантаження власного датасету, вам буде запропоновано вибрати директорію.

---

### Крок 2: Налаштування змінних для пошуку оптимальних параметрів

1. **Вибір кількості етапів пайплайну**  
   На новому екрані (Рисунок 4.2) встановіть кількість етапів (1–3), щоб визначити, скільки алгоритмів підряд буде застосовано для редукції.  

    ![Рисунок 4.2 – Форма вибору кількості алгоритмів](/media//pic2.png) <!-- Вставте посилання на зображення -->

2. **Вибір алгоритмів**  
   Для кожного етапу оберіть алгоритм зі списку: PCA, t-SNE, NMF, TruncatedSVD, FastICA, Autoencoder.

    ![Рисунок 4.3 – Вибір алгоритму редукції та параметрів до нього](/media/pic3.png) <!-- Вставте посилання на зображення -->

3. **Налаштування параметрів:**  
   - Мінімальний ступінь стиснення  
   - Максимальний ступінь стиснення  
   - Крок зміни параметра  

4. **Запуск тренування**  
   Натисніть кнопку **"Start"** для запуску процесу підбору оптимальних параметрів. Процес може тривати довго, залежно від обраних налаштувань.

---

### Крок 3: Аналіз результатів редукції

Після завершення тренування вам буде продемонстровано візуалізацію результатів у вигляді графіка (Рисунок 4.4).  
- **Оцінка результатів:** Оберіть найкращу комбінацію параметрів залежно від ваших потреб (наприклад, точність або швидкість).  

![Рисунок 4.4 – Приклад графіку з підібраними параметрами](/media/pic4.png) <!-- Вставте посилання на зображення -->

---

### Крок 4: Тренування на обраних параметрах

1. **Введення параметрів**  
   У вікні **"Навчання на власних параметрах"** (Рисунок 4.5) введіть вибрані параметри.  

    ![Рисунок 4.5 – Вікно для створення конвеєру на власних параметрах](/media/pic5.png) <!-- Вставте посилання на зображення -->

2. **Запуск тренування**  
   Натисніть кнопку **"Тренування"** для створення та навчання конвеєру з обраними параметрами. Отримані редуковані дані буде збережено у форматі `.csv`.  


3. **Вихід із форми**  
   Для завершення роботи натисніть **"Вихід"**.

---

## Висновок

Цей додаток дозволяє гнучко експериментувати з алгоритмами редукції даних, пропонуючи користувачам зручний інструмент для аналізу, візуалізації та збереження редукованих даних.  
