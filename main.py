import os
import csv
import argparse
import pandas as pd
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk import ngrams
from nltk.tokenize import word_tokenize

from config import *

STOP_WORDS_FROM_NLTK = set(stopwords.words('russian'))

PATH_BEST_N = str(os.path.join(FOLDER_SAVE, FOLDER_N_GRAMS))
PATH_FREQUENCIES = str(os.path.join(FOLDER_SAVE, FOLDER_FREQUENCY))
PATH_ABS = str(os.path.join(FOLDER_SAVE, FOLDER_FREQUENCY, CSV_ABS_FREQUENCY))
PATH_REL = str(os.path.join(FOLDER_SAVE, FOLDER_FREQUENCY, CSV_REL_FREQUENCY))


def create_dir(path):
    # Создание папки, если ее нет
    if not os.path.isdir(path):
        os.makedirs(path)


def create_csv(path):
    if not os.path.exists(path):
        data = {'texts': []}
        df = pd.DataFrame(data)
        df.to_csv(path, index=False, encoding='utf-8')


def create_dirs_for_results():
    # Создание папок для вывода результата

    create_dir(PATH_BEST_N)
    create_dir(PATH_FREQUENCIES)

    create_csv(PATH_ABS)
    create_csv(PATH_REL)


def extract_ngrams(text, n):
    # Создание n-грамм из текста
    words = [word for word in word_tokenize(text.lower()) if
             word.isalnum() and word not in STOP_WORDS and not word.isdigit() and
             word not in STOP_WORDS_FROM_NLTK]
    return list(ngrams(words, n))


def transform_grams(top_ngrams):
    return [(' '.join(ngram_tuple), count) for ngram_tuple, count in top_ngrams]


def count_ngrams(text, n_values, filename, text_ngram_counts, ngram_counts):
    """Подсчет n-грамм"""
    text_ngrams = {n: extract_ngrams(text, n) for n in n_values}
    text_ngram_count = {n: Counter(text_ngrams[n]) for n in n_values}
    text_ngram_counts.append((filename, text_ngram_count))

    for n in n_values:
        ngram_counts[n].update(text_ngram_count[n])

    return ngram_counts


def process_texts_from_directory(directory, n_values):
    """
    Открывает папку, проходит по всем файлам с форматом .txt,
    создает для каждой n свои n-граммы, считает количество и сохраняет
    :param directory: путь к папке
    :param n_values: кортеж для n в n_gram
    :return:
    """

    ngram_counts = {n: Counter() for n in n_values}

    text_ngram_counts = []

    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), encoding='utf-8') as file:
                text = file.read()
            n_grams = count_ngrams(text, n_values, filename,
                                   text_ngram_counts, ngram_counts)
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
            writer.writerows(transform_grams(top_ngrams))


def save_abs_freq_csv(ngram_counts, filename):
    """
    Сохранение n-грам со всез текстов
    :param ngram_counts:
    :param filename:
    :return:
    """

    def create_row():
        new_row = {'texts': filename}

        for n, counts in ngram_counts.items():
            for word, amount in counts.most_common(20):
                word = ' '.join(word)
                new_row[word] = amount

        return new_row

    df = pd.read_csv(PATH_ABS)

    column_texts = df['texts']

    if filename in column_texts.values:
        print(f'Файл {filename} уже существует в таблице.')
        return

    new_row_df = pd.DataFrame([create_row()])
    df = pd.concat([df, new_row_df], ignore_index=True).fillna(0)

    df.to_csv(PATH_ABS, index=False)


def save_rel_freq_csv():
    df = pd.read_csv(PATH_ABS)
    for col in df.columns[1:]:
        sum_col = df[col].sum()
        df[col] = df[col].apply(lambda x: x / sum_col)

    df.to_csv(PATH_REL, index=False)


def auto_nltk_tab():
    """
        Проверка на наличие и установка пакета nltk
        :return:
    """
    nltk.data.path.append(NTLK_DATA_DIRECTORY)
    if not os.path.exists(NTLK_DATA_DIRECTORY):
        os.makedirs(NTLK_DATA_DIRECTORY)

    try:
        nltk.data.find('tokenizers/stopwords')
    except LookupError:
        nltk.download('stopwords', download_dir=NTLK_DATA_DIRECTORY)


def is_file(file_path):
    if not os.path.exists(file_path):
        print(f"Файл {file_path} не существует.")
        quit(1)


def get_console():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Введите папку с файлами расширения .txt или файл.csv")
    args = parser.parse_args()
    is_file(args.input)
    return args.input


def main():
    auto_nltk_tab()

    input_file = get_console()

    create_dirs_for_results()

    if os.path.splitext(input_file)[1] == '':
        process_texts_from_directory(input_file, N_GRAMMS)
    else:
        print('Wrong dir/csv')
        quit(1)

    print('Successful')


if __name__ == '__main__':
    main()
