
import speech_recognition as sr
from sbert_punc_case_ru import SbertPuncCase

import torch
from aniemore.recognizers.voice import VoiceRecognizer
from aniemore.models import HuggingFaceModel

r = sr.Recognizer()
file_path = "audio.wav"
with sr.AudioFile(file_path) as source:
    audio_text = r.record(source)

text = r.recognize_google(audio_text, language="ru-RU")
#print(text)

model1 = SbertPuncCase()
model_per = model1.punctuate(text)
print(model_per)   #текст со знаками препинания


model = HuggingFaceModel.Voice.WavLM
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vr = VoiceRecognizer(model=model, device=device)
vr.recognize('audio.wav', return_single_label=True)

'''
from aniemore.recognizers.text import TextRecognizer
from aniemore.models import HuggingFaceModel

model = HuggingFaceModel.Text.Bert_Tiny2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tr = TextRecognizer(model=model, device=device)

tr.recognize('жалко что это работает? :(', return_single_label=True)
'''
