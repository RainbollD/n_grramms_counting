import csv
import argparse
import pandas as pd
from collections import Counter
import nltk
from nltk import ngrams
from nltk.tokenize import word_tokenize
import spacy

from config import *


def create_dir(path):
    # Создание папки, если ее нет
    if not os.path.isdir(path):
        os.makedirs(path)


def create_csv(path):
    if not os.path.exists(path):
        data = {'texts': []}
        df = pd.DataFrame(data)
        df.to_csv(path, index=False, encoding='utf-8')


def create_dirs():
    # Создание папок для вывода результата

    create_dir(PATH_BEST_N)
    create_dir(PATH_FREQUENCIES)

    create_csv(PATH_ABS)
    create_csv(PATH_REL)

    # Создание папок для дополнительных моделей

    create_dir(NTLK_DATA_DIRECTORY)
    create_dir(SPACY_DATA_DIRECTORY)


def extract_ngrams(text, n):
    # Создание n-грамм из текста
    words = [word for word in word_tokenize(text.lower()) if
             word.isalnum() and word not in STOP_WORDS and not word.isdigit() and
             word not in STOP_WORDS_FROM_NLTK]
    return list(ngrams(words, n))


def ngrams_to_str(top_ngrams):
    return [(' '.join(ngram_tuple), count) for ngram_tuple, count in top_ngrams]


def initial_word(text):
    """Переводит слова в тексте в начальную форму"""
    nlp = spacy.load(SPACY_DATA_DIRECTORY)
    doc = nlp(text)
    text = [token.lemma_ for token in doc]
    return ' '.join(text)


def count_ngrams(text, n_values, filename):
    """Подсчет n-грамм"""
    ngram_counts = {n: Counter() for n in n_values}
    text_ngram_counts = []

    text_ngrams = {n: extract_ngrams(text, n) for n in n_values}
    text_ngram_count = {n: Counter(text_ngrams[n]) for n in n_values}
    text_ngram_counts.append((filename, text_ngram_count))

    for n in n_values:
        ngram_counts[n].update(text_ngram_count[n])

    return ngram_counts


def process_texts_from_directory(directory, n_values=N_GRAMMS):
    """
    Открывает папку, проходит по всем файлам с форматом .txt,
    создает для каждой n свои n-граммы, считает количество и сохраняет
    :param directory: путь к папке
    :param n_values: кортеж для n в n_gram
    :return:
    """

    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), encoding='utf-8') as file:
                text = initial_word(file.read())

            n_grams = count_ngrams(text, n_values, filename)

            all_saving(n_grams, filename)


def all_saving(ngram_counts, filename):
    """Сохранение всех данных"""
    save_ngram_counts(ngram_counts, filename)
    save_abs_freq_csv(ngram_counts, filename)
    save_rel_freq_csv()


def save_ngram_counts(ngram_counts, filename):
    filename = filename.replace('.', '_')
    create_dir(os.path.join(PATH_BEST_N, filename))

    for n, counts in ngram_counts.items():
        top_ngrams = counts.most_common(20)
        with open(os.path.join(PATH_BEST_N, filename, f'top_{n}.csv'), 'w', newline='',
                  encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['n-gram', 'count'])
            writer.writerows(ngrams_to_str(top_ngrams))


def create_row(filename, ngram_counts):
    new_row = {'texts': filename}

    for n, counts in ngram_counts.items():
        for word, amount in counts.most_common(20):
            word = ' '.join(word)
            new_row[word] = amount

    return new_row


def is_file_in_cvs(filename, column_texts):
    if filename in column_texts.values:
        print(f'Файл {filename} уже существует в таблице.')
        return


def save_abs_freq_csv(ngram_counts, filename):
    """
    Сохранение n-грам со всез текстов
    :param ngram_counts:
    :param filename:
    :return:
    """

    df = pd.read_csv(PATH_ABS)

    column_texts = df['texts']

    is_file_in_cvs(filename, column_texts)

    new_row_df = pd.DataFrame([create_row(filename, ngram_counts)])
    df = pd.concat([df, new_row_df], ignore_index=True).fillna(0)

    df.to_csv(PATH_ABS, index=False)


def save_rel_freq_csv():
    df = pd.read_csv(PATH_ABS)
    for col in df.columns[1:]:
        sum_col = df[col].sum()
        df[col] = df[col].apply(lambda x: x / sum_col)

    df.to_csv(PATH_REL, index=False)


def auto_intall_models():
    def nltk_lab():
        """
            Проверка на наличие и установка пакета nltk
            :return:
        """
        nltk.data.path.append(NTLK_DATA_DIRECTORY)
        if not os.path.exists(NTLK_DATA_DIRECTORY):
            os.makedirs(NTLK_DATA_DIRECTORY)

        try:
            nltk.data.find('tokenizers/stopwords')
        except:
            nltk.download('stopwords', download_dir=NTLK_DATA_DIRECTORY)

    def spacy_lab():
        """
            Проверка на наличие и установка пакета spacy
            :return:
        """
        if len(os.listdir(SPACY_DATA_DIRECTORY)) == 0:
            try:
                nlp = spacy.load("ru_core_news_sm")
                nlp.to_disk(SPACY_DATA_DIRECTORY)
            except:
                print("Установите 'ru_core_news_sm'\npython -m spacy download ru_core_news_sm")

    nltk_lab()
    spacy_lab()


def is_it_folder(filename):
    if not os.path.splitext(filename)[1] == '':
        print("It isn't a folder")
        quit(1)


def is_file(file_path):
    if not os.path.exists(file_path):
        print(f"Файл {file_path} не существует.")
        quit(1)


def get_console():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Введите папку")
    args = parser.parse_args()
    is_file(args.input)
    return args.input


def main():
    create_dirs()

    auto_intall_models()

    input_file = 'tests'#get_console()

    is_it_folder(input_file)

    process_texts_from_directory(input_file)

    print('Successful')


if __name__ == '__main__':
    main()
