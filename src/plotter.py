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
import os

import util
from correlator import Correlator
from featurizer import FeaturizerManager

PLOTS_DIR = os.path.join(os.path.dirname(__file__), '../plots/')


class Plotter():
  """Make plots.

  Usage:
    plotter = Plotter()
    plotter.load_counts('ab_counts.pickle')
    plotter.plot_word_group_frequencies([
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


  def plot_word_group_frequencies(self, word_groups, stem=False,
    normalize=False):
    """Plot frequencies of word groups by era.

    Args:
      word_groups: array of arrays comprising word groups.
      stem (bool): whether to stem the input words.
      noralize (bool): whether to visualize percentage change instead of
        absolute change.

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

      if normalize:
        sum_frequencies = sum_frequencies / sum_frequencies[0]

      plt.plot(sum_frequencies,
        label='Group: {}'.format(word_group))

    plt.title('Word frequencies')
    if normalize:
      plt.ylabel('Word group frequency relative to start')
    else:
      plt.ylabel('Word group frequency')
    plt.xlabel('Era')
    plt.legend()
    plt.show()


class BookWcPlotter(Plotter):
  """Plot cohort frequencies by book
  """

  def plot_word_group_frequencies(self, word_groups):

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

def get_feature_value(featurizer_manager, feature_name, book_id, book_year):
  if feature_name is 'year':
    return book_year


def plot_features_by_name(x_feature, y_feature, pretty_x_name=None,
  pretty_y_name=None, save=False):
  if pretty_x_name is None:
    pretty_x_name = x_feature
  if pretty_y_name is None:
    pretty_y_name = y_feature

  logging.info('Plotting {} by {}'.format(y_feature, x_feature))

  fm = FeaturizerManager()
  x = []
  y = []
  for book_year, book_id in fm.get_book_year_and_ids():
    try:
      x.append(fm.get_feature_value_by_name(x_feature, book_id, book_year))
    except KeyError:
      logging.error('Invalid feature value for feature \'{}\' and book id \'{}\''
        .format(x_feature, book_id))
      x.append(0)
    try:
      y.append(fm.get_feature_value_by_name(y_feature, book_id, book_year))
    except KeyError:
      logging.error('Invalid feature value for feature \'{}\' and book id \'{}\''
        .format(y_feature, book_id))
      y.append(0)

  plt.scatter(x, y, color='green')

  # Line of best fit.
  plt.plot(x, np.poly1d(np.polyfit(x, y, 1))(x), '--', color='blue')

  plt.xlabel(pretty_x_name)
  plt.ylabel(pretty_y_name)
  plt.title(pretty_y_name)
  if save:
    filename = y_feature + '_by_' + x_feature + '.png'
    filepath = os.path.join(PLOTS_DIR, filename)
    logging.info('Saving plot: {}'.format(filepath))
    plt.savefig(filepath)
    plt.close()
  else:
    plt.show()

def plot_all_pos():
  all_pos = ['nouns', 'verbs', 'adjectives', 'adverbs', \
      'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', \
      'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', \
      'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', \
      'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']

  for pos in all_pos:
    plot_features_by_name('year', pos + ' all ratio',
      pretty_y_name='Proportion of part of speech: ' + pos, save=True)


def plot_features():
  plot_features_by_name('male_pronouns', 'female_pronouns')
  # plot_features_by_name('nouns', 'verbs')
  # plot_features_by_name('year', 'female_male_pronoun_ratio', save=True)
  # plot_features_by_name('year', 'word_count', save=True)
  # plot_features_by_name('year', 'type_token_ratio', save=True)
  # plot_features_by_name('year', 'vocab_size', save=True)
  # plot_features_by_name('year', 'nouns verbs ratio')
  # # plot_features_by_name('year', 'nouns adjectives ratio')
  # plot_features_by_name('year', 'nouns verbs ratio')
  # plot_features_by_name('year', 'nouns adjectives ratio')
  # plot_features_by_name('year', 'nouns adverbs ratio')
  # plot_features_by_name('year', 'nouns all ratio', pretty_y_name='Proportion of noun types')
  # plot_all_pos()


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

def make_frequency_plots_for_word_groups():
  c = Correlator()
  c.load_counts('data/pickle/tcc_counts_1900-1999.pickle')
  c.load_correlations('data/pickle/tcc_correlations_1900-1999.pickle')
  plotter = Plotter(correlator=c)

  # plotter.plot_word_group_frequencies([
  #   ['bitch', 'whore'],
  #   ['tattoo'],
  #   ['flint'],
  # ], normalize=True)

  # plotter.plot_word_group_frequencies([
  #   ['war'],
  #   ['damn'],
  #   ['sex', 'drug'],
  # ], normalize=True)

  # plotter.plot_word_group_frequencies([
  #   ['phone'],
  #   ['film'],
  #   ['letter'],
  # ], normalize=True)

  # plotter.plot_word_group_frequencies([
  #   ['onto'],
  #   ['of']
  # ], normalize=True)

  # plotter.plot_word_group_frequencies([
  #   ['which'],
  #   ['that']
  # ], normalize=True)

  plotter.plot_word_group_frequencies([
    ['one', 'two', 'three'],
    ['1', '2', '3']
  ], normalize=False)

def make_frequency_scatter_plots():
  c = Correlator()
  c.load_counts('data/pickle/tcc_wc_by_book.pickle')
  plotter = BookWcPlotter(correlator=c)

  plotter.plot_word_group_frequencies([
    ['one', 'two', 'three'],
    ['1', '2', '3']
  ])


def main():
  logging.basicConfig(format='[%(name)s %(asctime)s]\t%(msg)s',
    stream=sys.stderr, level=logging.DEBUG)
  # make_frequency_plots_by_era()
  # make_frequency_plots_for_word_groups()
  make_frequency_scatter_plots()

if __name__ == '__main__':
  main()
