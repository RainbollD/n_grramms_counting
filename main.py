import os
import csv
import argparse
import pandas as pd
from collections import Counter
from nltk import ngrams
from nltk.tokenize import word_tokenize

from config import *


def create_null_dir(path):
    # Создание папки, если ее нет
    if not os.path.isdir(path):
        os.makedirs(path)


def create_dirs_for_results():
    # Создание папок для вывода результата
    create_null_dir(FOLDER_SAVE)
    create_null_dir(os.path.join(FOLDER_SAVE, 'best_n'))
    create_null_dir(os.path.join(FOLDER_SAVE, 'frequencies'))


def is_file(file_path):
    if not os.path.exists(file_path):
        print(f"Файл {file_path} не существует.")
        quit(1)


def extract_ngrams(text, n):
    # Создание n-грамм из текста
    words = [word for word in word_tokenize(text.lower()) if
             word.isalnum() and word not in STOP_WORDS and not word.isdigit()]
    return list(ngrams(words, n))


def process_texts_from_directory(directory, n_values):
    """
    Открывает папку, проходит по всем файлам с форматом .txt,
    создает для каждой n свои n-граммы, считает количество и сохраняет
    :param directory: путь к папке
    :param n_values: кортеж для n
    :return:
    """
    ngram_counts = {n: Counter() for n in n_values}
    text_ngram_counts = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                text_ngrams = {n: extract_ngrams(text, n) for n in n_values}
                text_ngram_count = {n: Counter(text_ngrams[n]) for n in n_values}
                text_ngram_counts.append((filename, text_ngram_count))

                for n in n_values:
                    ngram_counts[n].update(text_ngram_count[n])

            save_ngram_counts(ngram_counts, filename)
            save_frequency_csv(text_ngram_counts, ngram_counts, filename)


def process_texts_from_csv(csv_file, n_values):
    """
    Открывает файл,
    создает для каждой n свои n-граммы, считает количество и сохраняет
    :param csv_file: путь к файлу .csv
    :param n_values: кортеж для n
    :return:
    """
    ngram_counts = {n: Counter() for n in n_values}
    text_ngram_counts = []
    df = pd.read_csv(csv_file)
    for index, row in df.iterrows():
        text = row.iloc[0]
        text_ngrams = {n: extract_ngrams(text, n) for n in n_values}
        text_ngram_count = {n: Counter(text_ngrams[n]) for n in n_values}
        text_ngram_counts.append((index, text_ngram_count))

        for n in n_values:
            ngram_counts[n].update(text_ngram_count[n])

    save_ngram_counts(ngram_counts, csv_file)
    save_frequency_csv(text_ngram_counts, ngram_counts, csv_file)


def save_ngram_counts(ngram_counts, filename):
    filename, type_ = os.path.splitext(filename)
    filename = f"{filename}_txt" if type_ == '.txt' else f"{filename}_csv"
    create_null_dir(os.path.join(FOLDER_SAVE, 'best_n', filename))

    for n, counts in ngram_counts.items():
        top_ngrams = counts.most_common(20)
        with open(os.path.join(FOLDER_SAVE, 'best_n', filename, f'top_{n}.csv'), 'w', newline='',
                  encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['n-gram', 'count'])
            writer.writerows(top_ngrams)


def save_frequency_csv(text_ngram_counts, ngram_counts, filename):
    filename, type_ = os.path.splitext(filename)
    filename = f"{filename}_txt" if type_ == '.txt' else f"{filename}_csv"
    create_null_dir(os.path.join(FOLDER_SAVE, 'frequencies', filename))

    absolute_freq = []
    relative_freq = []
    total_counts = {n: sum(count.values()) for n, count in ngram_counts.items()}

    """
    Добавить подсчет
    """

    # Сохранение абсолютной частоты
    with open(os.path.join(FOLDER_SAVE, 'frequencies', filename, f'absolute_frequency.csv'), 'w', newline='',
              encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['text_id'] + [f'top_{n}_ngram' for n in ngram_counts.keys()])
        writer.writerows(absolute_freq)

    # Сохранение относительной частоты
    with open(os.path.join(FOLDER_SAVE, 'frequencies', filename, f'relative_frequency.csv'), 'w', newline='',
              encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['text_id'] + [f'top_{n}_ngram' for n in ngram_counts.keys()])
        writer.writerows(relative_freq)


def console():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Введите папку с файлами расширения .txt или файл.csv")
    args = parser.parse_args()
    return args.input


def main():
    input_file = console()
    name_file, type_input_file = os.path.splitext(input_file)

    create_dirs_for_results()
    is_file(input_file)

    if type_input_file == '':
        process_texts_from_directory(input_file, N_GRAMMS)
    elif type_input_file == '.csv':
        process_texts_from_csv(input_file, N_GRAMMS)
    else:
        print('Wrong dir/csv')
        input()
        quit(1)

    print('Successful')


if __name__ == '__main__':
    main()
