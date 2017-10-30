#!/usr/bin/env python3
#
#    zeval.py
#    © 2017 Henning Gebhard <henning.gebhard@rub.de>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
zeval.py

Script to determine Craig's Zeta for a number of terms in a corpus
and evaluate its effectiveness for a text classifier.
"""

from datetime import datetime as dt
from glob import glob
from os.path import join, dirname, abspath
from random import sample

# NLP and data handling
import pandas as pd
from pandas import Series, DataFrame
import spacy
from spacy.language import Language

# Classification
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pygal as pg

##############################################################################
# Parameters

FILE_DIR = dirname(abspath(__file__))
CSV_FILE_NAME = 'zeta_specifity'
CORPUS_DIR = join(FILE_DIR, 'corpus')
SEGMENTS_LEN = 1000     # other sensible values could be 500, 1500, 2000
SEGMENT_LENGTHS = [100, 250, 500, 1000, 2000, 3000]
NLP = spacy.load('en')
CLASSIFIERS = [
    # LinearSVC,
    # SVC,                    # This one uses an rbf kernel by default
    KNeighborsClassifier,
    # MultinomialNB,
    GaussianNB,
    DecisionTreeClassifier,
    RandomForestClassifier
]
pd.set_option('display.max_colwidth', -1)


##############################################################################
# Utility functions

def extract_basename(filepath):
    filename = filepath.split('/')[-1]
    name_components = filename.split('.')[:-1]
    return "".join(name_components)


def chunks(it, n=SEGMENTS_LEN):
    """Yield successive n-sized chunks from it."""
    for i in range(0, len(it), n):
        yield it[i:i + n]

##############################################################################
#  Read files and tokenize them

def extract_features(text: str, nlp: Language) -> Series:
    """Iterate over a text and extract all words that occur."""
    tokens = [t.string for t in nlp.tokenizer(text)]
    tokens = [t.strip().lower() for t in tokens if t.strip().isalpha()]
    series = pd.Series(list(tokens))
    return series


def read_files(lang: Language, directory: str = CORPUS_DIR) -> DataFrame:
    """Read corpus text files and tokenize them.
    Each file is stored in a Series, with its tokens being a Series
    themselves. Every text combined is stored in a DataFrame.
    """
    frame = DataFrame(columns=['textid', 'author', 'tokens'])
    for f in glob(join(directory, '*.txt')):
        with open(f, 'r') as textfile:
            textid = extract_basename(f)
            data = {
                'tokens': extract_features(textfile.read(), lang),
                'textid': textid,
                'author': 'Poe' if textid.startswith('p') else 'Lovecraft',
            }
            series = Series(data)
            frame = frame.append(series, ignore_index=True)
    return frame

###################################################################################
#  Calculating zeta values

def build_vocab(frame: DataFrame) -> Series:
    """Extract all tokens from a DataFrame and generate a combined vocabulary."""
    text_series = frame.loc[:, 'tokens']
    vocab = pd.concat(list(text_series)).sort_values().drop_duplicates()
    return vocab


def build_tdm(frame: DataFrame, segment_length: int, vocab: Series) -> DataFrame:
    """Build a term document matrix.
    This function expects a DataFrame which contains an author column, an textid column and
    a tokens column which contains all the tokens of the corresponding text. It segments each
    text (separately) and determines for each segment, which term of the vocabulary it contains
    and which it does not.
    """
    tdm = DataFrame(index=vocab.values)
    tokens = frame['tokens']
    tids = frame['textid']
    for tokenlist, tid in zip(tokens, tids):
        segments = build_segments(tokenlist, segment_length)
        for segnu, segment in enumerate(segments):
            adjacency = vocab.isin(segment)
            adjacency.name = '{}seg{}'.format(tid, segnu)
            tdm[adjacency.name] = adjacency.values
    return tdm


def build_segments(text: Series, segment_length: int) -> Series:
    """Segment a text with a given segment lenght.
    Expects a Series of tokens, which represent a whole text. It throws away
    the last segment if it is smaller than 50% of the ideal segment length.
    """
    segments = chunks(text, segment_length)
    segments = [s.str.lower().str.strip() for s in segments]
    segments = Series(segments)
    if len(segments.iloc[-1]) < 0.5 * segment_length:
        segments = segments.drop(segments.index[-1])
    return segments


def calculate_zeta(text_frame: DataFrame):
    """Entry point to calculate zeta for all terms in the corpus."""
    # Build the complete vocabulary.
    vocab = build_vocab(text_frame)

    # Build a subcorpus T of only 'Poe' texts
    subcorpus_T = text_frame[text_frame['textid'].str.startswith('p')]
    # Build a subcorpus C of only 'Lovecraft' texts
    subcorpus_C = text_frame[text_frame['textid'].str.startswith('l')]

    # DataFrame to hold the specifity of each word in our vocabulary, according to Zeta with
    # a certain segment length.
    specifity = DataFrame(index=vocab.values)

    for segment_length in SEGMENT_LENGTHS:
        # Generate term document matrix with the segments as documents
        # for the current segment length.
        tdm_T = build_tdm(subcorpus_T, segment_length, vocab)
        tdm_C = build_tdm(subcorpus_C, segment_length, vocab)
        len_T = len(tdm_T.columns)
        len_C = len(tdm_C.columns)

        # Calculate Zeta for each term.
        spec_label = 'spec{}'.format(segment_length)
        specifity[spec_label] = (
            tdm_T.sum(axis=1) / len_T
            - tdm_C.sum(axis=1) / len_C
        )

    return vocab, specifity


##################################################################################################
#  Analyse results and classification

def specific_word_range(word_series):
    """Takes a series of term indexes which each contain most specific
    words for a certain segment length. We want to determine how large
    the whole vocabulary is for this series.
    Suppose that each index contains 6 terms. The whole vocab contains
    14 terms. This indicates a bigger variety than if the vocab
    of all indexes would only have a total of 10 words.
    We want to use the series of specific words with the highest vocab
    per regular index length.
    """
    all_words = word_series.iloc[0]
    for i in word_series.values:
        all_words = all_words.union(i)
    index_length = len(word_series.iloc[0])
    vocab_length = len(all_words)
    return (vocab_length, index_length)


def extract_most_specific(specifities: DataFrame, quantile=0.0005) -> Series:
    """Extract the most specific words for both subcorpora."""
    relevant_words = Series()

    for sm in SEGMENT_LENGTHS:
        label = 'spec' + str(sm)
        series = specifities[label]
        low_quantile = series.quantile(quantile)
        high_quantile = series.quantile(1-quantile)

        isSpec = 'isSpecFor' + str(sm)
        specifities[isSpec] = (specifities[label] < low_quantile) | (specifities[label] > high_quantile)
        words = specifities.loc[specifities[isSpec] == True].index
        relevant_words[label] = words

    return relevant_words


def calculate_rel_freq(text_frame, words):
    """Given a list of tokens from all texts, and a list of relevant words,
    determine the relative frequency of each relevant word for each text.
    """
    frequencies = DataFrame(index=words)
    tokenlists = text_frame['tokens']
    textids = text_frame['textid']

    for tokens, tid in zip(tokenlists, textids):
        text_len = len(tokens)
        frequencies[tid] = tokens.value_counts() / text_len

    frequencies = frequencies.fillna(0)
    return frequencies


def calculate_rel_frequencies(text_frame: DataFrame, words_list: Series) -> Series:
    """Determine the relative frequencies for the relevant words from
    all tested segment lengths.
    """
    frequencies = Series()
    for label, words in words_list.items():
        frequencies[label] = (calculate_rel_freq(text_frame, words))
    return frequencies


def evaluation(frequencies: Series, test_size) -> DataFrame:
    # The convention in scikit-learn is to use 'X' to denote the list
    # of feature vectors and 'y' to denote the list of target values
    # Do all this stuff for all segment lengths
    clf_names = [str(c).split('.')[-1][:-2] for c in CLASSIFIERS]
    clf_names.append('avg')
    results = DataFrame(index=clf_names)

    # label is the segment length, data the series of score values.
    for label, data in frequencies.items():
        # Initialize the column for this segment length with zeros.
        results[label] = None

        data = data.transpose()
        lovecraft_index = [label for label in data.index if label.startswith('l')]
        poe_index = [label for label in data.index if label.startswith('p')]
        poe_index = sample(poe_index, len(lovecraft_index))
        chosen = lovecraft_index + poe_index

        X = data.loc[chosen]
        y = [1 if textid.startswith('p') else -1 for textid in X.index]
        # Split data in trainings and test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        for clf_name, Classifier in zip(clf_names, CLASSIFIERS):
            clf = Classifier()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            results.loc[clf_name, label] = score

        results.loc['avg', label] = results.loc[:, label].mean()

    return results


def eval_significance(result_list):
    index = []
    clf_names = [str(c).split('.')[-1][:-2] for c in CLASSIFIERS]
    for c in clf_names:
        for sl in SEGMENT_LENGTHS:
            index.append(str(c) + str(sl))

    results = pd.DataFrame(index=index)

    for i, result in enumerate(result_list):
        results[i] = result_list[i].drop('avg').values.flatten()

    with open('results/complete_results.csv', 'w') as f:
        results.to_csv(f)
    return results


def multi_evaluation(frequencies, test_size=0.5):
    result_list = []
    for _ in range(200):
        result_list.append(evaluation(frequencies, test_size))

    eval_significance(result_list)

    classifiers = result_list[0].index
    seg_lengths = result_list[0].columns
    results = DataFrame(index=classifiers)

    all_averages = DataFrame()

    # Investigate each segment length
    for seg in seg_lengths:
        # From every result, take the columns with this segment length.
        columns = [data.loc[:, seg] for data in result_list]

        # A string to hold information about the spread of values
        div_string = "Segment length: {}\n###########################\n"

        # For each classifier, build the mean of all measured values
        # and add it to the final result.
        for clf in classifiers:
            values = [col.loc[clf] for col in columns]
            results.loc[clf, seg] = sum(values) / len(values)
            # Also, save the divergence of values for later analysis
            s = Series(values)
            div_string += "{}:\n{}\n".format(clf, s.describe().to_string())
            # If this is the 'avg' value of the current segment length, add it to its own df.
            if clf == 'avg':
                all_averages[seg] = s

        # Write info about the spread into a separate log file
        with open('results/div/div_{}.txt'.format(seg), 'w') as outfile:
            outfile.write(div_string)

    # Render a boxplot for each segment length's avg values
    render_boxplot(all_averages, 'results/div/all_averages.png')

    return results


#################################################################################################
# visualization stuff

def render_boxplot(specifities, filename='specifity_dist.png'):
    box = pg.Box()
    for segment_length in SEGMENT_LENGTHS:
        label = 'spec' + str(segment_length)
        box.add(label, specifities[label])
    box.render_to_png(filename)


def render_mean_performance(results, now):
    results = results.transpose()
    line = pg.Line(range=(0, 1), stroke=False)
    line.x_labels = results.index

    for label, data in results.items():
        stroke = label == 'avg'
        line.add(label, data.values, stroke=stroke)

    filename = "".join(['results/results', now, '.png'])
    line.render_to_png(filename)


##################################################################################################

def main():
    # Read all texts in the corpus directory and tokenize them.
    text_frame = read_files(NLP, CORPUS_DIR)
    vocab, specifities = calculate_zeta(text_frame)

    # Write specifities to CSV file.
    now = dt.now()
    date = "_{}_{}_{}_{}".format(now.month, now.day, now.hour, now.minute)
    filename = "".join(['results/', CSV_FILE_NAME, date, '.csv'])
    with open(filename, 'w') as outfile:
        specifities.to_csv(outfile)
    print('written specifities')
    render_boxplot(specifities)

    # Determine the most specific words.
    words = extract_most_specific(specifities)
    filename = "".join(['results/words', date, '.csv'])
    with open(filename, 'w') as outfile:
        words.to_csv(outfile)
    print('written specific words')

    # Determine their relative frequencies for each original text.
    frequencies = calculate_rel_frequencies(text_frame, words)

    # Make a number of evaluations
    results = multi_evaluation(frequencies)

    # Write summary of results into text file.
    filename = "".join(['results/results', date, '.csv'])
    with open(filename, 'w') as outfile:
        results.to_csv(outfile)
    render_mean_performance(results, date)
    print('written performance results')

    return vocab, results


if __name__ == '__main__':
    main()
