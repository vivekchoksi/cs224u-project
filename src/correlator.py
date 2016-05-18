#!/usr/bin/python

"""This module implements a Correlator tool as described by The Stanford Literary Lab.

http://litlab.stanford.edu/LiteraryLabPamphlet4.pdf
"""

import os
import operator
import numpy as np
import logging
import sys
import cPickle as pickle
from collections import defaultdict
from scipy.stats.stats import pearsonr


DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/')

class ExampleReader():
  """Read data from the corpus by "era". 

  Eras could represent decades or any other grouping of documents in the
  corpus.
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
          for word in line.split():
            yield word

  def get_num_eras(self):
    # Hard-coded example.
    return 2

  def _get_filenames(self, era):
    # Hard-coded example.
    if era == 0:
      filenames = ['example0.txt']
    elif era == 1:
      filenames = ['example1.txt']
    else:
      raise ValueError('Invalid era.')

    return [os.path.join(DATA_DIR, f) for f in filenames]



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

  def preprocess(self, reader, k):
    self._count_occurrences(reader)
    self._correlate_words(k)
    # TODO: Store self.correlations as a pickle file once we have actual data.

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
      for word in reader.read_era(era):
        counts[word][era] += 1
        counts_by_era[era] += 1

    # Normalize by total word count in each era.
    for word in counts:
      counts[word] /= counts_by_era

    self.counts = counts

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
    for w1 in self.counts:
      for w2 in self.counts:
        if w1 is not w2:
          correlations[w1][w2] = pearsonr(self.counts[w1], self.counts[w2])[0]

      # To save space, only store top k correlations. That is, for each w1,
      # only store k w2. The trade-off is that there is redundant computation.
      correlations[w1] = sorted(correlations[w1].items(),
        key=operator.itemgetter(1), reverse=True)[:k]

      self.correlations = correlations


def main():
    logging.basicConfig(format='[%(name)s %(asctime)s]\t%(msg)s',
      stream=sys.stderr, level=logging.DEBUG)
    correlator = Correlator()
    correlator.preprocess(reader=ExampleReader(), k=2)
    print correlator.get_cohort('f')

if __name__ == '__main__':
    main()
