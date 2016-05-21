#!/usr/bin/python

"""Utility functions.
"""

import cPickle as pickle
import logging

def pickle_load(pickle_filename):
  logging.info('Reading from pickle file: \'{}\'...'.format(pickle_filename))
  return pickle.load(open(pickle_filename, 'rb'))

def pickle_dump(data, pickle_filename):
  logging.info('Dumping to pickle file: \'{}\'...'.format(pickle_filename))
  pickle.dump(data, open(pickle_filename, 'wb'))
