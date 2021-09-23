""" This file is used for creating an overview of the information contained in all the different sessions of the UN
speeches

Authors:

"""

import csv
import os
import sys
import re
import nltk

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from tqdm import tqdm
from nltk.sentiment import SentimentIntensityAnalyzer


# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('vader_lexicon')

try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

dir_path = os.path.dirname(os.path.realpath(__file__))
main_data_dir = os.path.join(dir_path, 'TXT')


def plot_sentiment_country_vs_year(country_code):
    """
    This function plots the sentiment scores of a specific country over the years
    :param country_code: the code of the country that you want the sentiment development plot for
    :return:
    """

    plt.figure()
    plt.title('{} speech sentiment'.format(country_code))
    plt.plot(speeches_df.loc[speeches_df['country'] == country_code]['year'],
             speeches_df.loc[speeches_df['country'] == country_code]['pos_sentiment'], label='positive')
    plt.plot(speeches_df.loc[speeches_df['country'] == country_code]['year'],
             speeches_df.loc[speeches_df['country'] == country_code]['neu_sentiment'], label='neutral')
    plt.plot(speeches_df.loc[speeches_df['country'] == country_code]['year'],
             speeches_df.loc[speeches_df['country'] == country_code]['neg_sentiment'], label='negative')
    plt.legend()
    plt.xlabel('Time (year)', fontsize=14)
    plt.ylabel('Sentiment', fontsize=14)
    plt.savefig('{}_sentiment_development.png'.format(country_code), dpi=300)
    plt.show()


def plot_correlation_matrix(speeches_df, corr_cols):
    """
    This function creates a correlation matrix of specific columns from the speeches df
    :param speeches_df: dataframe containing all the information about the speeches
    :param corr_cols: a list of columns that we cant to calculate the correlation for
    :return:
    """

    plt.figure()
    sns.heatmap(speeches_df.loc[:, corr_cols].corr(),
                annot = True, vmin=-1, vmax=1, center= 0)
    plt.show()

def count_most_used_words(data, n):
    """
    This functions a list of the n most used words with their occurrence rate
    :param data: the string to gather the word occurrences from
    :param n: number of most used words in the speeches
    :return: two lists, first with the most occurring words and then their occurrence rate
    """

    occ_words = Counter(data).most_common(n)

    words = [occ_tup[0] for occ_tup in occ_words]
    counts = [occ_tup[1] for occ_tup in occ_words]

    return words, counts


def count_specific_words(data, words):
    """
    This function creates a list of the occurrences of a specific set of words
    :param data: the string to gather the word occurrences from
    :param words: the words to look for in the string
    :return: a list containing the occurrences of the words in order
    """

    occ_words = Counter(data)
    occ_words_in_question = [occ_words[word] for word in words]
    return occ_words_in_question

def count_total_words(data):
    """ This function purely counts the number of words a piece of text

    :param file: a piece of text to count the number of words from
    :return: the word count
    """

    word_count = len(data)

    return word_count

def determine_sentiment(data):
    """
    This function determines the sentiment of a string (in this case a speech)
    :param data: the speech / a string
    :return:
    """

    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(data)

    return sentiment


def open_speech(file_path):
    """
    This function opens a file with the correct formatting
    :param file_path:
    :return:
    """

    file = open(file_path, encoding="utf-8-sig")
    data = file.read()

    return data




def preprocess_speech(data):
    """
    This function does the preprocessing
    :param data:
    :return:
    """

    # put all characters in lower case
    data = data.lower()

    # only keep the tokens of the data
    data = nltk.word_tokenize(data)

    # remove stop words and non-alphabetic stuff from all the text
    sw = nltk.corpus.stopwords.words("english")
    no_sw = np.empty(len(data), dtype = str)

    i = 0
    for w in data:
        if (w not in sw) and w.isalpha():
            no_sw[i] = w
            i += 1

    no_sw = np.trim_zeros(no_sw)

    return no_sw


if __name__ == '__main__':

    # True --> run preprocessing and save the results, False --> just do the data analysis with your previously saved
    # dataframe file (always have to do a preprocessing run to save the dataframe of course)
    do_preprocessing = False

    if do_preprocessing:
        speeches_df = pd.DataFrame(columns=['session_nr', 'year', 'country', 'word_count', 'pos_sentiment',
                                            'neu_sentiment', 'neg_sentiment'])

        num_directories = len(next(os.walk(main_data_dir))[1])

        # loop through all directories of the data
        for root, subdirectories, files in tqdm(os.walk(main_data_dir), total=num_directories, desc='directory: '):

            # remove all the files starting with '.' (files created by opening a mac directory on a windows PC,
            # so will only do something if you are working on a windows PC
            files_without_dot = [file for file in files if not file.startswith('.')]

            # loop through files and extract data
            for file in tqdm(files_without_dot, desc='files: ', leave=False):
                country, session_nr, year = file.replace('.txt', '').split('_')

                # open a speech with the correct formatting
                speech_data = open_speech(os.path.join(root, file))

                # preprocess the data
                preprocessed_bag_of_words = preprocess_speech(speech_data)

                # calculate all the features through functions
                word_count = count_total_words(preprocessed_bag_of_words)
                most_used_words = count_most_used_words(preprocessed_bag_of_words, 20)
                occs_of_spec_words = count_specific_words(preprocessed_bag_of_words, ['economy'])
                sentiment_of_speech = determine_sentiment(speech_data)

                # append the line of features to the dataframe
                speeches_df = speeches_df.append({'session_nr': int(session_nr),
                                                  'year': int(year),
                                                  'country':country,
                                                  'word_count':word_count,
                                                  'pos_sentiment': sentiment_of_speech['pos'],
                                                  'neu_sentiment': sentiment_of_speech['neu'],
                                                  'neg_sentiment': sentiment_of_speech['neg']
                                                  },
                                                 ignore_index=True)

        # add the country names to the dataframe
        df_codes = pd.read_csv('UNSD — Methodology.csv', delimiter=',')
        speeches_df = speeches_df.merge(df_codes, how='left', left_on='country', right_on='ISO-alpha3 Code')

        # add the happiness dataframe to the
        happiness_df = pd.read_excel('DataPanelWHR2021C2.xls', index_col=[0, 1])
        speech_happi_merged_df = pd.merge(speeches_df, happiness_df, how='left',
                                          left_on=['year', 'Country or Area'], right_on=['year', 'Country name'])

        speech_happi_merged_df.to_csv('preprocessed_dataframe.csv')



    # get into the exploration after reading the saved file
    speeches_df = pd.read_csv('preprocessed_dataframe.csv')


    corr_cols = ['year', 'word_count', 'pos_sentiment', 'neg_sentiment', 'neu_sentiment']
    plot_correlation_matrix(speeches_df, corr_cols)

    # plot some figures from the data
    plot_sentiment_country_vs_year('NLD')
    plot_sentiment_country_vs_year('USA')
