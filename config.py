import os
from nltk.corpus import stopwords

STOP_WORDS_FROM_NLTK = set(stopwords.words('russian'))

FOLDER_SAVE = 'result_ngramms'
FOLDER_N_GRAMS = 'best_n'
FOLDER_FREQUENCY = 'frequencies'
CSV_ABS_FREQUENCY = 'absolute_frequency.csv'
CSV_REL_FREQUENCY = 'relative_frequency.csv'

PATH_BEST_N = str(os.path.join(FOLDER_SAVE, FOLDER_N_GRAMS))
PATH_FREQUENCIES = str(os.path.join(FOLDER_SAVE, FOLDER_FREQUENCY))
PATH_ABS = str(os.path.join(FOLDER_SAVE, FOLDER_FREQUENCY, CSV_ABS_FREQUENCY))
PATH_REL = str(os.path.join(FOLDER_SAVE, FOLDER_FREQUENCY, CSV_REL_FREQUENCY))


N_GRAMMS = tuple((i for i in range(1, 21)))

STOP_WORDS = ["в", "к", "с", "на", "о", "об", "обо", "во", "ко", "со", "по", "под", "за", "до", "над", "от",
              "ото", "пред", "при", "у", "вне", "из", "про", "для", "без",
              "он", "его", "него", "ему", "нему", "им", "ним", "она", "ее", "нее", "ей", "ней", "оно", "мне", "я",
              "моя", "мой", "меня", "мной", "моем", "моём", "мои", "моих", "моим", "они", "их", "них", "им", "ним",
              "вы", "вам", "вас", "вами", "ваш", "ваша", "ваше", "вашего", "вашей", "ваши", "вашим", "вашему",
              "ты", "тебе", "тобой", "тебя", "твой", "твоя", "твои", "твоих", "твоим", "твоей", "твоего", "твоими",
              "себе", "себя", "собой", "свой", "своя", "свои", "своих", "своим", "своего", "своей", "мы", "нас",
              "наш", "наша", "наши", "нашего", "нашим", "наших", "нашу", "наше", "нашей", "нашему", "либо", "после",
              "сквозь", "вокруг",
              "*", "-", "—", "–", "были"]

NTLK_DATA_DIRECTORY = 'nltk_lab'
