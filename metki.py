import ast
import pandas as pd

# Загрузка токенизированного датасета
tokenized_data_path = 'tokenized_train_dataset.csv'
tokenized_data = pd.read_csv(tokenized_data_path)

# Просмотр первых нескольких строк датасета для понимания структуры данных
tokenized_data_head = tokenized_data.head()
def simple_entity_tagger(tokenized_text):
    entities = []
    for token in tokenized_text:
        if token.isupper() and len(token) > 1:  # Пример простого правила для обнаружения аббревиатур
            entities.append((token, 'ABBREVIATION'))
        elif token.istitle():  # Слова с заглавной буквы могут быть именами собственными
            entities.append((token, 'PROPER_NOUN'))
        else:
            entities.append((token, 'O'))  # 'O' означает отсутствие сущности
    return entities

# Применяем функцию к каждой строке
tokenized_data['Entities'] = tokenized_data['Tokenized Text'].apply(lambda x: simple_entity_tagger(ast.literal_eval(x)))

# Сохраняем результат
tokenized_data.to_csv('tokenized_with_entities.csv', index=False)
