#!/usr/bin/python

"""Utility functions.
"""

import string
import logging
import cPickle as pickle
from nltk.stem.snowball import SnowballStemmer

STEMMER = SnowballStemmer("english", ignore_stopwords=True)

def pickle_load(pickle_filename):
  logging.info('Reading from pickle file: \'{}\'...'.format(pickle_filename))
  return pickle.load(open(pickle_filename, 'rb'))

def pickle_dump(data, pickle_filename):
  logging.info('Dumping to pickle file: \'{}\'...'.format(pickle_filename))
  pickle.dump(data, open(pickle_filename, 'wb'))

def tokenize_words(line):
  """Stem and tokenize the words in a line of raw text.
  """
  # Lower case and remove punctuation.
  fmt_line = line.lower().replace('--', ' ').replace(',', ' ')
  fmt_line = fmt_line.translate(string.maketrans('', ''), string.punctuation)

  # Stem words.
  fmt_line = fmt_line.decode('utf-8')
  return [STEMMER.stem(w) for w in fmt_line.split()]

def fn_to_title(fpath):
  filename = fpath.split("/")[-1]
  return filename.split(",")[0]

