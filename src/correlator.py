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
    end_era = min(self.START_YEAR + (era + 1) * self.era_size,
      self.END_YEAR + 1)

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
    correlator.preprocess_counts(reader=Reader())
    correlator.get_cohort('hard') # Will return after some computation.
    correlator.preprocess_correlations(k=20)
    correlator.get_cohort('hard') # Will return immediately.

  Attributes:
    counts: a dict mapping from word to numpy array of that word's normalized
      frequency by era.
    correlations: a dict mapping from word to list of k tuples of the form
      (word, Pearson correlation coefficient).
  """
  def __init__(self):
    self.counts = None
    self.correlations = None

  def preprocess_counts(self, reader, pickle_dump_file=None):
    """Count word frequencies in the corpus.

    Args:
      reader: an instance of a Reader object.
      pickle_dump_file (Optional[str]): name of pickle file to which to
        write word occurrence count data.
    """
    self._count_occurrences(reader)
    self._prune_counts()
    if pickle_dump_file is not None:
      util.pickle_dump(self.counts, pickle_dump_file)

  def load_counts(self, pickle_load_file):
    self.counts = util.pickle_load(pickle_load_file)

  def preprocess_correlations(self, k=20, pickle_dump_file=None):
    """Precompute correlations between word pairs.

    Args:
      k (int): the maximum number of top correlated words to keep track of for
        each word.
      pickle_dump_file (Optional[str]): name of pickle file to which
        to write computed word correlations.
    """
    self._correlate_words(k)
    if pickle_dump_file is not None:
      util.pickle_dump(self.correlations, pickle_dump_file)

  def load_correlations(self, pickle_load_file):
    self.correlations = util.pickle_load(pickle_load_file)

  def get_cohort(self, word, k=20):
    """Given a word, return the group of words in its semantic cohort.

    Args:
      word (str)

    Returns:
      list: a list of tuples of the form
        (word, Pearson correlation coefficient).
    """
    if self.counts is None:
      raise Exception('Counts not yet computed. Must call ' +
        'preprocess_counts() or load_counts().')
    elif self.correlations is not None and word in self.correlations:
      # The cohort is precomputed.
      return self.correlations[word][:k]
    else:
      # Compute the cohort now.
      return self._correlate_word(word, k)

  def _count_occurrences(self, reader):
    """Calculate normalized word frequencies for each word by era as
    self.counts.

    Args:
      reader: an instance of a Reader object.
    """
    logging.info('Counting occurrences of words...')
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

  def _prune_counts(self):
    logging.info('Discarding rare words...')
    original_length = len(self.counts)
    for word in self.counts.keys():
      # Only keep words that were present in every era.
      if 0 in self.counts[word]:
        del self.counts[word]
    logging.info('\tVocabulary size reduced from {} to {}'.format(
      original_length, len(self.counts)))

  def _correlate_words(self, k):
    """Calculate similar word pairs based on their frequency counts as
    self.correlations.

    Similarity is calculated using the Pearson correlation coefficient.

    Args:
      k (int): the maximum number of top correlated words to keep track of for
        each word.
    """
    if self.counts is None:
      raise Exception(
        'Must calculate word counts first, by calling count_occurrences().')

    logging.info('Calculating correlations between words...')
    correlations = defaultdict(lambda: {})

    counter = 0

    # NOTE(vivekchoksi): Might be faster if this operation were vectorized;
    # that is, compute Pearson coefficients using matrices instead of a loop.
    for w1 in self.counts:
      correlations[w1] = self._correlate_word(w1, k)

      counter += 1
      if counter % 10 == 0:
        logging.info('\tFinished correlating {} of {} words...'.format(
          counter, len(self.counts)))

      if counter == 150:
        break

    self.correlations = dict(correlations)

  def _correlate_word(self, word, k):
    # Correlations for `word`. E.g. correlations[word2] = Pearson coefficient
    correlations = {}

    for word2 in self.counts:
      if word is not word2:
        correlations[word2] = pearsonr(self.counts[word], self.counts[word2])[0]

    # To save space, only store top k correlations. That is, for each w1,
    # only store k w2. The trade-off is that there is redundant computation.
    return sorted(correlations.items(), key=operator.itemgetter(1), reverse=True)[:k]


def run_correlator():
  correlator = Correlator()
  # correlator.preprocess_counts(AmericanBestsellersReader(), 'ab_counts.pickle')
  correlator.load_counts('ab_counts.pickle')
  # correlator.preprocess_correlations(k=20, pickle_dump_file='ab_correlations.pickle')
  # correlator.load_correlations('ab_correlations.pickle')
  print correlator.get_cohort('great')
  pdb.set_trace()

def main():
  logging.basicConfig(format='[%(name)s %(asctime)s]\t%(msg)s',
    stream=sys.stderr, level=logging.DEBUG)
  run_correlator()

if __name__ == '__main__':
  main()
