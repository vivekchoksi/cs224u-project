#!/usr/bin/python

"""This module implements a tool to plot frequencies of words or groups of
words over time.
"""

import sys
import logging
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import util
from correlator import Correlator
from featurizer import FeaturizerManager


class Plotter():
  """Make plots.

  Usage:
    plotter = Plotter()
    plotter.load_counts('ab_counts.pickle')
    plotter.plot_cohort_frequencies([
      ['war', 'shortage', 'casualties'],
      ['thou', 'didst', 'hast']
    ])

  Attributes:
    counts: a dict mapping from word to numpy array of that word's normalized
      frequency by era.
  """

  def __init__(self, counts=None, correlator=None):
    self.counts = counts
    self.correlator = correlator
    if correlator is not None:
      self.counts = correlator.counts

  def load_counts(self, pickle_load_file):
    self.counts = util.pickle_load(pickle_load_file)

  def plot_cohort_frequencies(self, seeds, stem=False):
    if self.correlator is None:
      self.correlator = Correlator()
      self.correlator.counts = self.counts

    if stem:
      seeds = util.stem_words(seeds)

    word_groups = []
    for w in seeds:
      word_groups.append([w] +
        [cohort_word for cohort_word, _ in self.correlator.get_cohort(w)])

    self.plot_word_group_frequencies(word_groups)


  def plot_word_group_frequencies(self, word_groups, stem=False):
    """Plot frequencies of word groups by era.

    Args:
      word_groups: array of arrays comprising word groups.
      stem (bool): whether to stem the input words.

    Usage:
      plotter.plot_word_group_frequencies([
        ['war', 'shortage', 'casualties'],
        ['thou', 'didst', 'hast']
      ])
    """
    print 'Word groups...'
    for word_group in word_groups:
      if stem:
        word_group = util.stem_words(word_group)

      print '\t', word_group

      # Sum frequencies of all words in the group.
      for i in range(len(word_group)):
        if i == 0:
          sum_frequencies = np.copy(self.counts[word_group[0]])
        else:
          sum_frequencies += self.counts[word_group[i]]

      plt.plot(sum_frequencies,
        label='Cohort for \'{}\''.format(word_group[0]))

    plt.title('Word frequencies')
    plt.ylabel('Normalized word frequency')
    plt.xlabel('Era')
    plt.legend()
    plt.show()


class BookWcPlotter(Plotter):
  """Plot cohort frequencies by book
  """

  def plot_cohort_frequencies(self, word_groups):

    title_to_data = self.counts

    colors = cm.rainbow(np.linspace(0, 1, len(word_groups)))
    # for each word group
    for word_group, c in zip(word_groups, colors):

      word_group = util.stem_words(word_group)

      years = []
      normalized_wcs = []

      #for each book
      for title, data in title_to_data.iteritems():
        year, counter, total_wc = data

        raw_wc = sum([counter[word] for word in word_group])
        norm_wc = raw_wc / float(total_wc)

        years.append(year)
        normalized_wcs.append(norm_wc)

      plt.scatter(years, normalized_wcs, color=c, label='Cohort for \'{}\''.format(word_group[0]))

    plt.title('Word frequencies by Novel')
    plt.ylabel('Normalized word frequency')
    plt.xlabel('Year')
    plt.legend()
    plt.show()


def plot_features_by_name(x_feature, y_feature):
  fm = FeaturizerManager()
  x = []
  y = []
  for book_id in fm.get_book_ids():
    try:
      x.append(fm.get_characteristic(x_feature, book_id))
    except:
      x.append(0)
    try:
      y.append(fm.get_characteristic(y_feature, book_id))
    except:
      y.append(0)

  plt.scatter(x, y)
  plt.xlabel(x_feature)
  plt.ylabel(y_feature)
  plt.show()

def plot_features():
  plot_features_by_name('male_pronouns', 'female_pronouns')

def make_frequency_plots_by_era():
  c = Correlator()
  c.load_counts('data/pickle/tcc_counts_1900-1999.pickle')
  c.load_correlations('data/pickle/tcc_correlations_1900-1999.pickle')
  plotter = Plotter(correlator=c)

  fluctuation_fn = c.upward_fluctuation
  cohorts = [w for w, _ in c.get_most_fluctuating_cohorts(fluctuation_fn)[:5]]
  plotter.plot_cohort_frequencies(cohorts)

  fluctuation_fn = c.downward_fluctuation
  cohorts = [w for w, _ in c.get_most_fluctuating_cohorts(fluctuation_fn)[:5]]
  plotter.plot_cohort_frequencies(cohorts)


def main():
  logging.basicConfig(format='[%(name)s %(asctime)s]\t%(msg)s',
    stream=sys.stderr, level=logging.DEBUG)
  # make_frequency_plots_by_era()
  plot_features()

if __name__ == '__main__':
  main()
