# README

# Проект Анализа Обращений Граждан

Данный проект посвящен анализу и классификации текстовых обращений граждан с использованием методов машинного обучения. Основная цель проекта - автоматизация процесса категоризации обращений и выделения ключевых сущностей для улучшения процессов обработки и реагирования.

# Содержание
1. [Структура проекта](#структура-проекта)
2. [Подготовка данных](#подготовка-данных)
3. [Модель Gradient Boosting Classifier и Random Forest Classifier](#модель-gradient-boosting-classifier-и-random-forest-classifier)
4. [Модель на основе Трансформера](#модель-на-основе-трансформера)
5. [Требования](#требования)
6. [Запуск проекта](#запуск-проекта)
7. [Результаты](#результаты)

# Структура проекта
- **DataWork**: Включает все модели и программы для обработки текста и его нормализации.
- **Итоги моделей**: Здесь находятся все полученные файлы на выходе у моделей.
- **Модель без Трансформеров и Модель на Трансформерах**: Два отдельных каталога, содержащих модели, реализованные соответственно без использования трансформеров и на основе трансформеров.
- **Киллер-фичи**: Здесь находятся все доп. модели и функции: определение эмоциональной окраски текста, работа с голосовыми сообщениями, обработка детально текста и т.д.

# Подготовка данных
Данные обращений граждан предварительно очищены и нормализованы. Предобработка включает удаление специальных символов, HTML тегов, лемматизацию и удаление стоп-слов. После очистки данные токенизируются для дальнейшего анализа.

# Модель Gradient Boosting Classifier и Random Forest Classifier
Используется для классификации текстов по темам и группам тем. Эта комбинация моделей обеспечивает высокую точность и устойчивость к переобучению.

# Модель на основе Трансформера
Применяется для более глубокого анализа текстов с помощью передовых алгоритмов на базе трансформеров. Эта модель способна более точно понимать контекст и нюансы языка.
### Предобработка предоставленного датасета

Содержится в [блокноте]("notebooks/preprocessing.ipynb").

### Обучение первого этапа

Содержится в [блокноте]("notebooks/cls2.ipynb").
Обучались 3 классифкатора для категорий "Исполнитель", "Группа тем", "Тема".


# Требования
Необходимые библиотеки:
- pandas
- scikit-learn
- matplotlib
- nltk
- tqdm
- sklearn_crfsuite

# Запуск проекта
Для запуска проекта выполните следующие шаги:
1. Установите зависимости из [Требований](#требования).
2. Загрузите предобработанные данные обращений граждан.
3. Запустите скрипт классификации для категоризации текстов.
4. Используйте модель CRF для выделения сущностей.

# Результаты
Результаты обучения моделей на тестовой подвыборке:

**Gradient Boosting Classifier и Random Forest Classifier:**
- Группа тем: 0.7809
- Тема: 0.4106


**Модель на основе Трансформера:**
- Исполнитель: 0.8095
- Группа тем: 0.8021
- Тема: 0.5321


---

*Проект обеспечивает точную и быструю классификацию обращений, учитывая как традиционные методы машинного обучения, так и передовые технологии на основе трансформеров.*
