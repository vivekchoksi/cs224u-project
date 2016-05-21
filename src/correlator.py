#!/usr/bin/python

"""This module implements a Correlator tool as described by The Stanford Literary Lab.

http://litlab.stanford.edu/LiteraryLabPamphlet4.pdf
"""

import os
import operator
import numpy as np
import logging
import sys
import string
import pdb
import cPickle as pickle
from collections import defaultdict
from scipy.stats.stats import pearsonr

import util

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/')


class Reader():
  """Read data from the corpus by "era". 

  Eras could represent decades or any other grouping of documents in the
  corpus.

  This class is for example and uses hard-coded example data. Subclasses are
  expected to read actual corpuses.
  """
  def __init__(self):
    pass

  def read_era(self, era):
    """Given an era, yield words in the corpus from that era.

    Args:
      era (int)

    Yields:
      str: a word in the corpus.
    """
    for file in self._get_filenames(era):
      with open(file, 'r') as infile:
        for line in infile:
          # Make lowercase, remove punctuation, and split into words.
          for word in line.lower().translate(
            string.maketrans('', ''), string.punctuation).split():
            yield word

  def get_num_eras(self):
    # Hard-coded example.
    return 2

  def _get_filenames(self, era):
    # Hard-coded example.
    if era == 0:
      filenames = ['ebooks-unzipped/1895/12190-8.txt']
    elif era == 1:
      filenames = ['ebooks-unzipped/1923/1156.txt']
    else:
      raise ValueError('Invalid era.')

    return [os.path.join(DATA_DIR, f) for f in filenames]


class AmericanBestsellersReader(Reader):
  """Read data from American Bestsellers list as provided by Project Gutenberg.
  https://www.gutenberg.org/wiki/Bestsellers,_American,_1895-1923_(Bookshelf)
  """
  START_YEAR = 1895
  END_YEAR = 1923

  def __init__(self, era_size=5):
    """TODO: Comment
    """
    self.era_size = era_size

    # Count "incomplete eras" (e.g. round up if the time period contains
    # some number of eras plus a fraction of an era).
    self.num_eras = int(np.ceil(
      float(self.END_YEAR - self.START_YEAR + 1) / era_size))

  def get_num_eras(self):
    return self.num_eras

  def _get_filenames(self, era):
    if era < 0 or era >= self.num_eras:
      raise ValueError('Invalid era.')

    start_era = self.START_YEAR + era * self.era_size
    end_era = self.START_YEAR + (era + 1) * self.era_size

    filenames = []
    corpus_dir = os.path.join(DATA_DIR, 'ebooks-unzipped')
    for year in range(start_era, end_era):
      year_dir = os.path.join(corpus_dir, str(year))
      for filename in os.listdir(year_dir):
        filenames.append(os.path.join(year_dir, filename))

    return filenames


class Correlator():
  """Calculate semantic cohorts for words based on frequency counts by era.

  Usage:
    correlator = Correlator()
    correlator.preprocess(reader=ExampleReader(), k=20)
    correlator.get_cohort('hard')

  Attributes:
    counts: a dict mapping from word to numpy array of that word's normalized
      frequency by era.
    correlations: a dict mapping from word to list of k tuples of the form
      (word, Pearson correlation coefficient).
  """
  def __init__(self):
    self.counts = None
    self.correlations = None

  def preprocess(self, reader=None, k=20, counts_pickle_to_load=None,
    correlations_pickle_to_load=None, counts_pickle_to_dump=None,
    correlations_pickle_to_dump=None):
    """Preprocess the data by counting word occurrences and computing
    correlations between words, or by loading these values from pickle files.

    Args:
      reader (Optional): an instance of a Reader object.
      k (int): the maximum number of top correlated words to keep track of for
        each word.
      counts_pickle_to_load (Optional[str]): name of pickle file storing word
        occurrence counts. Either this or `reader` must be specified.
      correlations_pickle_to_load (Optional[str]): name of pickle file storing
        correlations between words.
      counts_pickle_to_dump (Optional[str]): name of pickle file to which to
        write computed word occurrence counts.
      correlations_pickle_to_dump (Optional[str]): name of pickle file to which
        to write computed word correlations.
    """
    # Get word counts, either by reading the corpus or loading a pickle file.
    if counts_pickle_to_load is None:
      if reader is None:
        raise ValueError(
          'Must either specify a reader or a counts pickle file.')
      else:
        self._count_occurrences(reader)
        if counts_pickle_to_dump is not None:
          util.pickle_dump(self.counts, counts_pickle_to_dump)
    else:
      self.counts = util.pickle_load(counts_pickle)

    # Get word correlations, either by reading from word counts or by loading
    # a pickle file.
    if correlations_pickle_to_load is None:
      self._correlate_words(k)
      if correlations_pickle_to_dump is not None:
        util.pickle_dump(self.correlations, correlations_pickle_to_dump)
    else:
      self.correlations = util.pickle_load(correlations_pickle)

  def get_cohort(self, word):
    """Given a word, return the group of words in its semantic cohort.

    Args:
      word (str)

    Returns:
      list: a list of tuples of the form
        (word, Pearson correlation coefficient).
    """
    if self.correlations is None:
      raise Exception('Must preprocess first by calling preprocess().')
    return self.correlations[word]

  def _count_occurrences(self, reader):
    """Calculate normalized word frequencies for each word by era as
    self.counts.

    Args:
      reader: an instance of a Reader object.
    """
    logging.info('Counting occurences of words...')
    num_eras = reader.get_num_eras()

    counts = defaultdict(lambda: np.zeros(num_eras))
    counts_by_era = np.zeros(num_eras)

    # Count all word occurrences.
    for era in range(num_eras):
      logging.info('\tReading era {} of {}...'.format(era + 1, num_eras))
      for word in reader.read_era(era):
        counts[word][era] += 1
        counts_by_era[era] += 1

    # Normalize by total word count in each era.
    for word in counts:
      counts[word] /= counts_by_era

    self.counts = dict(counts)

  def _correlate_words(self, k):
    """Calculate similar word pairs based on their frequency counts as
    self.correlations.

    Similarity is calculated using the Pearson correlation coefficient.

    Args:
      k (int): the maximum number of top correlated words to keep track of for
        each word.
    """
    if self.counts is None:
      raise Exception('Must calculate word counts first, by calling count_occurrences().')

    logging.info('Calculating correlations between words...')

    correlations = defaultdict(lambda: {})

    print 'Vocabulary size:', len(self.counts)
    # for word in self.counts:
    #   print word, self.counts[word]
    counter = 0
    
    for w1 in self.counts:
      for w2 in self.counts:
        if w1 is not w2:
          correlations[w1][w2] = pearsonr(self.counts[w1], self.counts[w2])[0]

      counter += 1
      if counter % 10 == 0:
        logging.info('Correlating...')
        print 'Finished correlating', counter, 'of', len(self.counts), 'words...'

      # To save space, only store top k correlations. That is, for each w1,
      # only store k w2. The trade-off is that there is redundant computation.
      correlations[w1] = sorted(correlations[w1].items(),
        key=operator.itemgetter(1), reverse=True)[:k]

    self.correlations = dict(correlations)

def run_correlator():
    correlator = Correlator()
    correlator.preprocess(reader=Reader(), k=20, counts_pickle_to_dump='example_counts.pickle',
    correlations_pickle_to_dump='example_correlations.pickle')
    print correlator.get_cohort('great')
    pdb.set_trace()

def main():
    logging.basicConfig(format='[%(name)s %(asctime)s]\t%(msg)s',
      stream=sys.stderr, level=logging.DEBUG)
    run_correlator()

if __name__ == '__main__':
    main()
