import pandas as pd
import re
import nltk
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

file_path = 'train_dataset_train.csv'

try:
    data = pd.read_csv(file_path, on_bad_lines='skip')
except Exception as e:
    load_error = str(e)
    data = None

if data is not None:
    data_info = data.info()
    data_head = data.head()
    data_describe = data.describe(include='all')
    load_error = None
else:
    data_info, data_head, data_describe = None, None, None

print(load_error, data_info, data_head, data_describe)

