import os
import csv
import argparse
import pandas as pd
from collections import Counter
from nltk import ngrams
from nltk.tokenize import word_tokenize
import nltk

nltk.download('stopwords')

STOP_WORDS = ["так", "то", "вот", "уже", "ни", "же", "не", "бы", "а", "там", "чтоб", "ли", "ли", "б", "хоть", "уж"],
FOLDER_SAVE = 'result_ngramms'
N_GRAMMS = tuple((i for i in range(1, 3)))

def extract_ngrams(text, n):
    words = [word for word in word_tokenize(text.lower()) if word.isalnum() and word not in STOP_WORDS and not word.isdigit()]
    return list(ngrams(words, n))


def process_texts_from_directory(directory, n_values):
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

    return ngram_counts, text_ngram_counts


def process_texts_from_csv(csv_file, n_values):
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

    return ngram_counts, text_ngram_counts


def save_ngram_counts(ngram_counts):
    for n, counts in ngram_counts.items():
        top_ngrams = counts.most_common(20)
        with open(os.path.join(FOLDER_SAVE, 'best_n', f'top_{n}.csv'), 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['n-gram', 'count'])
            writer.writerows(top_ngrams)


def save_frequency_tables(text_ngram_counts, ngram_counts):
    absolute_freq = []
    relative_freq = []
    total_counts = {n: sum(count.values()) for n, count in ngram_counts.items()}

    for text_id, ngram_count in text_ngram_counts:
        abs_row = []
        rel_row = []
        for n in ngram_counts.keys():
            for ngram in ngram_counts[n].keys():
                abs_row.append(ngram_count[n][ngram])
                rel_row.append(ngram_count[n][ngram] / total_counts[n] if total_counts[n] > 0 else 0)
        absolute_freq.append([text_id] + abs_row)
        relative_freq.append([text_id] + rel_row)

    # Сохранение абсолютной частоты
    with open(os.path.join(FOLDER_SAVE, 'frequencies', f'absolute_frequency.csv'), 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['text_id'] + [f'top_{n}_ngram' for n in ngram_counts.keys()])
        writer.writerows(absolute_freq)

    # Сохранение относительной частоты
    with open(os.path.join(FOLDER_SAVE, 'frequencies', f'relative_frequency.csv'), 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['text_id'] + [f'top_{n}_ngram' for n in ngram_counts.keys()])
        writer.writerows(relative_freq)


def console():
    parser = argparse. ArgumentParser(description="Простое CLI-приложение на Python")
    parser.add_argument("input", help="Введите папку с файлами расширения .txt или файл.csv")
    args = parser.parse_args()
    return args.input


def main():
    console()
    type_input_file = console()

    ngram_counts = {}
    text_ngram_counts = []

    if type_input_file == 'directory':
        ngram_counts, text_ngram_counts = process_texts_from_directory('tests', N_GRAMMS)
    elif type_input_file == 'csv':
        ngram_counts, text_ngram_counts = process_texts_from_csv('test_text.csv', N_GRAMMS)

    save_ngram_counts(ngram_counts)
    save_frequency_tables(text_ngram_counts, ngram_counts)


if __name__ == '__main__':
    main()
