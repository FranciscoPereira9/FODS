""" This file is used for creating an overview of the information contained in all the different sessions of the UN
speeches

Authors:

"""

import csv
import os

import numpy as np
import pandas as pd


dir_path = os.path.dirname(os.path.realpath(__file__))
main_data_dir = os.path.join(dir_path, 'TXT')

speeches_df = pd.DataFrame(columns=['session_nr', 'year', 'country', 'word_count'])


def count_words(data):
    """ This function purely counts the number of words a piece of text

    :param file: a piece of text to count the number of words from
    :return: the word count
    """

    words = data.split()
    word_count = len(words)

    return word_count


def open_speech(file_path):
    """
    This function opens a file with the correct formatting
    :param file_path:
    :return:
    """

    file = open(file_path, encoding="utf8")
    data = file.read()

    return data


def preprocess_speech(data):

    return data


if __name__ == '__main__':

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

            print(year)
            print(speech_data[0:100])

            break

            # preprocess the data
            preprocessed_speech = preprocess_speech(speech_data)

            # calculate all the features through functions
            word_count = count_words(preprocessed_speech)

            # append the line of features to the dataframe
            speeches_df = speeches_df.append({'session_nr': session_nr, 'year': year, 'country':country, 'word_count':
                                             word_count}, ignore_index=True,)

    print(speeches_df)
