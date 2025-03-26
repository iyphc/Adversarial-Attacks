# Adversarial Attacks on CIFAR-10 with CNN

Проект реализует DeepFool и RGD состязательные атаки для сверточной нейронной сети CNN, обученной на датасете CIFAR-10. Включает обучение модели, оценку ее устойчивости к атакам и расчет доверительных интервалов.

## Содержание
1. [Установка](#установка)
2. [Структура проекта](#структура-проекта)
3. [Использование](#использование)
4. [Примеры](#примеры)

---

## Установка

**Зависимости:**
- Python
- PyTorch
- torchvision
- numpy
- scipy
- tqdm


1. Создайте и активируйте виртуальное окружение:
   ```bash
   python -m venv venv
   source venv/bin/activate      # Для Linux/MacOS
   venv\Scripts\activate         # Для Windows
   ```
2. Установите зависимости: 
    ```bash
    pip install -r requirements.txt
    ```

---

## Структура проекта

├── attacks.py     
├── data.py        
├── model.py       
├── train.py       
├── utils.py       
├── evaluation.py  
├── .gitignore     
└── README.md      
