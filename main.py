import os
import csv
import argparse
import pandas as pd
from collections import Counter
from nltk import ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

stop_words = set(stopwords.words('russian'))


def extract_ngrams(text, n):
    words = [word for word in word_tokenize(text.lower()) if word.isalnum() and word not in stop_words]
    return list(ngrams(words, n))


def process_texts_from_directory(filename, n_values):
    ngram_counts = {n: Counter() for n in n_values}
    text_ngram_counts = []

    if filename.endswith('.txt'):
        with open(filename, 'r', encoding='utf-8') as file:
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
        text = row[0]
        text_ngrams = {n: extract_ngrams(text, n) for n in n_values}
        text_ngram_count = {n: Counter(text_ngrams[n]) for n in n_values}
        text_ngram_counts.append((index, text_ngram_count))

        for n in n_values:
            ngram_counts[n].update(text_ngram_count[n])

    return ngram_counts, text_ngram_counts


def save_ngram_counts(ngram_counts, output_prefix):
    for n, counts in ngram_counts.items():
        top_ngrams = counts.most_common(20)
        with open(f'{output_prefix}_top_{n}.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['n-gram', 'count'])
            writer.writerows(top_ngrams)


def save_frequency_tables(text_ngram_counts, ngram_counts, output_prefix):
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
    with open(f'{output_prefix}_absolute_frequency.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['text_id'] + [f'top_{n}_ngram' for n in ngram_counts.keys()])
        writer.writerows(absolute_freq)

    # Сохранение относительной частоты
    with open(f'{output_prefix}_relative_frequency.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['text_id'] + [f'top_{n}_ngram' for n in ngram_counts.keys()])
        writer.writerows(relative_freq)


def main_console():
    parser = argparse.ArgumentParser(description='Extract n-grams from text files and analyze their frequency.')
    parser.add_argument('--input', required=True, help='Input directory for text files or CSV file.')
    parser.add_argument('--type', choices=['directory', 'csv'], required=True, help='Type of input.')
    parser.add_argument('--n', type=int, nargs='+', required=True, help='Values of n for n-grams (e.g., 1 2 3).')
    parser.add_argument('--output', required=True, help='Prefix for output CSV files.')
    args = parser.parse_args()

    n_values = args.n
    output_prefix = args.output

    if args.type == 'directory':
        ngram_counts, text_ngram_counts = process_texts_from_directory(args.input, n_values)
    elif args.type == 'csv':
        ngram_counts, text_ngram_counts = process_texts_from_csv(args.input, n_values)

    save_ngram_counts(ngram_counts, output_prefix)
    save_frequency_tables(text_ngram_counts, ngram_counts, output_prefix)


def main():
    n_gramms = tuple((i for i in range(1, 21)))
    type_input_file = 'directory'
    if type_input_file == 'directory':
        ngram_counts, text_ngram_counts = process_texts_from_directory('test.txt', n_gramms)
    elif type_input_file == 'csv':
        ngram_counts, text_ngram_counts = process_texts_from_csv('test.txt', n_gramms)

    save_ngram_counts(ngram_counts, 'end.csv')
    save_frequency_tables(text_ngram_counts, ngram_counts, 'end.csv')


if __name__ == '__main__':
    main()
