""" This file is used for creating an overview of the information contained in all the different sessions of the UN
speeches

Authors:

"""

import csv
import os
import sys
import re

import numpy as np
import pandas as pd

from collections import Counter

dir_path = os.path.dirname(os.path.realpath(__file__))
main_data_dir = os.path.join(dir_path, 'TXT')


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

    data = data.lower()

    return data


if __name__ == '__main__':

    speeches_df = pd.DataFrame(columns=['session_nr', 'year', 'country', 'word_count'])

    # loop through all directories of the data
    for root, subdirectories, files in os.walk(main_data_dir):

        # remove all the files starting with '.' (files created by opening a mac directory on a windows PC, so will only
        # do something if you are working on a windows PC
        files_without_dot = [file for file in files if not file.startswith('.')]

        # loop through files and extract data
        for file in files_without_dot:
            country, session_nr, year = file.replace('.txt', '').split('_')

            # open a speech with the correct formatting
            speech_data = open_speech(os.path.join(root, file))

            # preprocess the data
            preprocessed_speech = preprocess_speech(speech_data)
            preprocessed_bag_of_words = preprocessed_speech.split(' ')

            # calculate all the features through functions
            word_count = count_total_words(preprocessed_bag_of_words)
            most_used_words = count_most_used_words(preprocessed_bag_of_words, 20)
            occs_of_spec_words = count_specific_words(preprocessed_bag_of_words, ['economy'])

            # append the line of features to the dataframe
            speeches_df = speeches_df.append({'session_nr': session_nr, 'year': year, 'country':country, 'word_count':
                                             word_count}, ignore_index=True)

    print(speeches_df)
