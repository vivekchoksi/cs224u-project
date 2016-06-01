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

def tokenize_words(line, stem=True):
  """Stem and tokenize the words in a line of raw text.
  """
  # Lower case and remove punctuation.
  fmt_line = line.lower().replace('--', ' ').replace(',', ' ')
  fmt_line = fmt_line.translate(string.maketrans('', ''), string.punctuation)

  # Different files may be in different encodings.
  try:
    fmt_line = fmt_line.decode('utf-8')
  except UnicodeDecodeError:
    fmt_line = fmt_line.decode('iso-8859-1')

  # Stem words before returning.
  if stem:
    return stem_words(fmt_line.split())
  else:
    return fmt_line.split()


def stem_words(words):
  """given list of word strings, return list of stemmed words
  """
  return [STEMMER.stem(w) for w in words]

def fn_to_title(fpath):
  filename = fpath.split("/")[-1]
  return "".join(filename.split(",")[:-1])

def filename_to_book_id(filename):
  """Return the first characters of the book that constitute numbers.
  """
  book_id = ''
  for char in filename:
    if is_number(char):
      book_id += char
    else:
      return book_id

def is_number(char):
  try:
    float(char)
    return True
  except ValueError:
    return False
