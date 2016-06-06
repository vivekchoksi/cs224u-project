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
from collections import defaultdict

import util
from correlator import Correlator
from featurizer import FeaturizerManager

PLOTS_DIR = os.path.join(os.path.dirname(__file__), '../plots/')
BOOK_METADATA = os.path.join(os.path.dirname(__file__), '../data/pickle/book_metadata.pickle')

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

  def plot_word_group_frequencies(self, word_groups, stem=False):

    title_to_data = self.counts

    colors = cm.rainbow(np.linspace(0, 1, len(word_groups)))

    max_y = 0.0

    # for each word group
    for word_group, c in zip(word_groups, colors):

      if stem:
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

      max_y = max(max_y, np.max(normalized_wcs))

      plt.scatter(years, normalized_wcs,
        color=c, label='Group: {}'.format(word_group), alpha=0.5)

    # Set the scaling to show all y values nicely.
    plt.ylim([-max_y / 15.0, 1.2 * max_y])
    # plt.yscale('log')
    
    plt.title('Word frequencies by novel')
    plt.ylabel('Normalized word frequency')
    plt.xlabel('Year')
    plt.legend()
    plt.show()

def make_groupings(book_metadata, group_by_attr):
  groupings = defaultdict(lambda: [])
  fm = FeaturizerManager()
  for book_id in fm.get_book_ids(book_metadata):
    attr_val = book_metadata[book_id][group_by_attr].strip()
    groupings[attr_val].append(book_id)
  return groupings

<<<<<<< HEAD
def make_list_groupings(book_metadata):
  lists = ['exp_rank', 'pub_prog_rank', 'ml_readers_rank', 'ml_editors_rank', 'bestsellers_rank']
  groupings = {list_name: [] for list_name in lists}
  fm = FeaturizerManager()
  for book_id in fm.get_book_ids(book_metadata):
    best_list = None
    best_rank = sys.maxint
    for list_name in groupings:
      curr_rank = book_metadata[book_id][list_name]
      if curr_rank > 0 and curr_rank < best_rank:
        best_list = list_name
        best_rank = curr_rank
    # print book_id, book_metadata[book_id], best_list
    assert(best_list is not None)
    groupings[best_list].append(book_id)
  return groupings
=======




"""
list_group1, list_group2: list of lists
i.e., list_group1 = ['ml_readers_rank', 'ml_editors_rank']


Default: assigns every book to a single list, based on originally established 
rules for handling overlaps

If list_group1 and list_group2 given: assigns every book to either the first or
second group of lists.  If overlap=True, creates a third list for books that are in
both of the list groups; otherwise, ignores books on both lists.
(Excludes books not on either of the list groups.)

"""

def make_list_groupings(book_metadata, list_group1=None, list_group2=None, overlap=True):

  # default, original case
  if not (list_group1 and list_group2):
    lists = ['exp_rank', 'pub_prog_rank', 'ml_readers_rank', 'ml_editors_rank', 'bestsellers_rank']
    groupings = {list_name: [] for list_name in lists}
    fm = FeaturizerManager()
    for book_id in fm.get_book_ids():
      best_list = None
      best_rank = sys.maxint
      for list_name in groupings:
        curr_rank = book_metadata[book_id][list_name]
        if curr_rank > 0 and curr_rank < best_rank:
          best_list = list_name
          best_rank = curr_rank
      # print book_id, book_metadata[book_id], best_list
      assert(best_list is not None)
      groupings[best_list].append(book_id)
    return groupings

  # list groups provided by the user
  else:
    list1_key = ",".join(list_group1)
    list2_key = ",".join(list_group2)
    groupings = {list1_key:[], list2_key:[],}
    if overlap:
      groupings['overlap'] = []

    fm = FeaturizerManager()

    #for every book_id
    for book_id in fm.get_book_ids():
      md = book_metadata[book_id]

      in_group1 = False
      in_group2 = False

      #check membership in group1
      for book_list in list_group1:
        if md[book_list] > 0:
          in_group1 = True

      #check membership in group2
      for book_list in list_group2:
        if md[book_list] > 0:
          in_group2 = True

      #assign to appropriate group
      if in_group1 and in_group2 and overlap:
        groupings['overlap'].append(book_id)
      elif in_group1:
        groupings[list1_key].append(book_id)
      elif in_group2:
        groupings[list2_key].append(book_id)

    return groupings

>>>>>>> 9a8036b0bab5fc2a329a71da7b411c0aab4f1a40

def plot_features_by_name_and_group(book_metadata, groupings,
  x_feature, y_feature, pretty_x_name=None,
  pretty_y_name=None, save=False):
  if pretty_x_name is None:
    pretty_x_name = x_feature
  if pretty_y_name is None:
    pretty_y_name = y_feature

  logging.info('Plotting {} by {}'.format(y_feature, x_feature))

  colors = cm.rainbow(np.linspace(0, 1, len(groupings)))
  attr_to_color = {}
  for attr_val, color in zip(groupings.keys(), colors):
    attr_to_color[attr_val] = color

  fm = FeaturizerManager()
  for attr_val in groupings:
    x = []
    y = []
    for book_id in groupings[attr_val]:
      book_year = book_metadata[book_id]['year']
      curr_x = None
      curr_y = None
      try:
        curr_x = fm.get_feature_value_by_name(x_feature, book_id, book_year)
      except KeyError:
        logging.error('Invalid feature value for feature \'{}\' and book id \'{}\''
          .format(x_feature, book_id))
      try:
        curr_y = fm.get_feature_value_by_name(y_feature, book_id, book_year)
      except KeyError:
        logging.error('Invalid feature value for feature \'{}\' and book id \'{}\''
          .format(y_feature, book_id))
      
      if curr_x is not None and curr_y is not None:
        x.append(curr_x)
        y.append(curr_y)

    plt.scatter(x, y, color = attr_to_color[attr_val], label=attr_val, alpha=0.5, s=np.pi*15)

  plt.xlabel(pretty_x_name)
  plt.ylabel(pretty_y_name)
  plt.title(pretty_y_name)
  plt.legend()

  # Shrink plot width by 20%.
  ax = plt.gca()
  box = ax.get_position()
  ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

  # Put a legend to the right of the current axis.
  ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


  if save:
    filename = y_feature + '_by_' + x_feature + '.png'
    filepath = os.path.join(PLOTS_DIR, filename)
    logging.info('Saving plot: {}'.format(filepath))
    plt.savefig(filepath)
    plt.close()
  else:
    plt.show()


def plot_features_by_name(book_metadata, x_feature, y_feature, pretty_x_name=None,
  pretty_y_name=None, save=False):
  if pretty_x_name is None:
    pretty_x_name = x_feature
  if pretty_y_name is None:
    pretty_y_name = y_feature

  logging.info('Plotting {} by {}'.format(y_feature, x_feature))

  fm = FeaturizerManager()
  x = []
  y = []
  for book_year, book_id in fm.get_book_year_and_ids(book_metadata):
    curr_x = None
    curr_y = None
    try:
      curr_x = fm.get_feature_value_by_name(x_feature, book_id, book_year)
    except KeyError:
      logging.error('Invalid feature value for feature \'{}\' and book id \'{}\''
        .format(x_feature, book_id))
    try:
      curr_y = fm.get_feature_value_by_name(y_feature, book_id, book_year)
    except KeyError:
      logging.error('Invalid feature value for feature \'{}\' and book id \'{}\''
        .format(y_feature, book_id))

    if curr_x is not None and curr_y is not None:
      x.append(curr_x)
      y.append(curr_y)
      plt.scatter(curr_x, curr_y)

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
  book_metadata = util.pickle_load(BOOK_METADATA)
  groupings = make_groupings(book_metadata, 'gender')
  list_groupings = make_list_groupings(book_metadata)

  all_pos = ['nouns', 'verbs', 'adjectives', 'adverbs', \
      'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', \
      'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', \
      'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', \
      'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB'
  ]

  for pos in all_pos:
    plot_features_by_name_and_group(book_metadata, list_groupings, 'year', pos + ' all ratio',
      pretty_y_name='Proportion of part of speech: ' + pos, save=False)


def plot_features():
  book_metadata = util.pickle_load(BOOK_METADATA)
  groupings = make_groupings(book_metadata, 'gender')
  #list_groupings = make_list_groupings(book_metadata)
  list_groupings = make_list_groupings(book_metadata, ['ml_editors_rank'], ['bestsellers_rank'])
  # plot_features_by_name_and_group(book_metadata, list_groupings, 'year', 'female_male_pronoun_ratio')

  # plot_features_by_name_and_group(book_metadata, groupings, 'year', 'median_sentence_length')
  # plot_features_by_name('year', 'median_sentence_length')
  # plot_features_by_name_and_group(book_metadata, list_groupings, 'year', 'flesch_kincaid')

  # plot_features_by_name('year', 'flesch_kincaid')

  # plot_all_pos()

  plot_features_by_name_and_group(book_metadata, list_groupings, 'male_pronouns', 'female_pronouns')
  # plot_features_by_name('nouns', 'verbs')
  # plot_features_by_name('year', 'word_count', save=True)
  # plot_features_by_name('year', 'type_token_ratio', save=True)
  # plot_features_by_name_and_group(book_metadata, list_groupings, 'year', 'type_token_ratio', save=False)
  # plot_features_by_name(book_metadata, 'year', 'nouns verbs ratio')
  # # plot_features_by_name(book_metadata, 'year', 'nouns adjectives ratio')
  # plot_features_by_name(book_metadata, 'year', 'nouns verbs ratio')
  # plot_features_by_name(book_metadata, 'year', 'nouns adjectives ratio')
  # plot_features_by_name(book_metadata, 'year', 'nouns adverbs ratio')
  # plot_features_by_name(book_metadata, 'year', 'nouns all ratio', pretty_y_name='Proportion of noun types')


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

  plotter.plot_word_group_frequencies([
    ['bitch', 'whore'],
    ['tattoo'],
    ['flint'],
  ], normalize=True)

  plotter.plot_word_group_frequencies([
    ['war'],
    ['damn'],
    ['sex', 'drug'],
  ], normalize=True)

  plotter.plot_word_group_frequencies([
    ['phone'],
    ['film'],
    ['letter'],
  ], normalize=True)

  plotter.plot_word_group_frequencies([
    ['onto'],
    ['of']
  ], normalize=True)

  plotter.plot_word_group_frequencies([
    ['which'],
    ['that']
  ], normalize=True)

  plotter.plot_word_group_frequencies([
    ['one', 'two', 'three'],
    ['1', '2', '3']
  ], normalize=False)

def make_frequency_scatter_plots():
  c = Correlator()
  c.load_counts('data/pickle/tcc_wc_by_book.pickle')
  # c.load_counts('data/pickle/wc_by_book.pickle')
  plotter = BookWcPlotter(correlator=c)

  plotter.plot_word_group_frequencies([
    ['phone'],
    ['film'],
    ['letter'],
  ])

  plotter.plot_word_group_frequencies([
    ['bitch', 'whore'],
    ['tattoo'],
    ['flint'],
  ])

  plotter.plot_word_group_frequencies([
    ['war'],
    ['damn'],
    ['sex', 'drug'],
  ])

def main():
  logging.basicConfig(format='[%(name)s %(asctime)s]\t%(msg)s',
    stream=sys.stderr, level=logging.DEBUG)
  plot_features()
  # make_frequency_plots_by_era()
  # make_frequency_plots_for_word_groups()
  # make_frequency_scatter_plots()

if __name__ == '__main__':
  main()
