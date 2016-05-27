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

  def __init__(self, counts=None):
    self.counts = counts

  def load_counts(self, pickle_load_file):
    self.counts = util.pickle_load(pickle_load_file)

  def plot_cohort_frequencies(self, word_groups, stem=False):
    """Plot frequencies of word groups by era.

    Args:
      word_groups: array of arrays comprising word groups.
      stem (bool): whether to stem the input words.

    Usage:
      plotter.plot_cohort_frequencies([
        ['war', 'shortage', 'casualties'],
        ['thou', 'didst', 'hast']
      ])
    """
    for word_group in word_groups:
      if stem:
        word_group = util.stem_words(word_group)

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


def main():
  logging.basicConfig(format='[%(name)s %(asctime)s]\t%(msg)s',
    stream=sys.stderr, level=logging.DEBUG)
  plotter = Plotter()
  plotter.load_counts('data/pickle/tcc_counts.pickle')
  # plotter.plot_cohort_frequencies([
  #   ['war', 'shortage', 'casualties'],
  #   ['thou', 'didst', 'hast']
  # ])
  plotter.plot_cohort_frequencies([
    [u'me', u'gentl', u'my', u'i', u'claus', u'humor', u'taylor', u'delici', u'steepl', u'submiss', u'enterpris', u'bind', u'misde', u'grey', u'groan', u'lip', u'christian', u'succumb', u'steel', u'shackl', u'wideey'],
    [u'she', u'her', u'herself', u'perfect', u'incap', u'convict', u'an', u'habit', u'gentleman', u'convent', u'refin', u'rather', u'afraid', u'prevent', u'reminisc', u'generos', u'inspir', u'much', u'pretend', u'to', u'present'],
    [u'shell', u'young', u'that', u'societi', u'suppos', u'phase', u'prevent', u'genius', u'daughter', u'her', u'afraid', u'convent', u'delic', u'refin', u'rather', u'herself', u'attribut', u'she', u'comprehens', u'incap', u'admir'],
    [u'delic', u'nevertheless', u'miseri', u'littl', u'fix', u'artifici', u'poor', u'afraid', u'that', u'piti', u'shell', u'made', u'generous', u'solemn', u'gentlemen', u'ought', u'better', u'devot', u'great', u'her', u'which'],
    [u'lucki', u'top', u'back', u'on', u'behind', u'stop', u'crack', u'kick', u'pick', u'mess', u'neck', u'stomach', u'glass', u'four', u'watch', u'sound', u'start', u'around', u'off', u'nowher', u'mayb'],
    [u'gentl', u'grey', u'submiss', u'humor', u'groan', u'flush', u'taylor', u'enterpris', u'christian', u'claus', u'delici', u'soft', u'lip', u'me', u'uncomfort', u'scold', u'steel', u'shyli', u'impass', u'inner', u'my']
  ])

if __name__ == '__main__':
  main()
