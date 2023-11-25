import pandas as pd
import re
import nltk
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

file_path = 'train_dataset_train.csv'
try:
    data = pd.read_csv(file_path, sep=';', on_bad_lines='skip')
except Exception as e:
    print(f"Ошибка при загрузке файла: {e}")
    exit()

nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('russian'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    try:
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^а-яА-ЯёЁ\s]', '', text)
        return ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    except Exception as e:
        print(f"Ошибка при обработке текста: {e}")
        return text

def tokenize_text(text):
    try:
        return text.split()
    except Exception as e:
        print(f"Ошибка при токенизации текста: {e}")
        return []

tokenized_data = pd.DataFrame(columns=['ID', 'Tokenized Text'])

for i in tqdm(range(len(data)), desc="Обработка данных"):
    try:
        cleaned_text = clean_text(data['Текст инцидента'].iloc[i])
        data['Текст инцидента'].iloc[i] = cleaned_text
        tokenized_row = pd.DataFrame({'ID': [i], 'Tokenized Text': [tokenize_text(cleaned_text)]})
        tokenized_data = pd.concat([tokenized_data, tokenized_row], ignore_index=True)

        if i % 1000 == 0:
            data.to_csv('processed_train_dataset_partial.csv', index=False)  # для себя, ошибок много было, смотрел где!
            tokenized_data.to_csv('tokenized_train_dataset_partial.csv', index=False)
    except Exception as e:
        print(f"Ошибка на строке {i}: {e}")

data.to_csv('processed_train_dataset.csv', index=False)
tokenized_data.to_csv('tokenized_train_dataset.csv', index=False)
