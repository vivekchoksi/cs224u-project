#!/usr/bin/python

"""This module implements a tool to plot frequencies of words or groups of
words over time.
"""

import sys
import logging
import matplotlib
import matplotlib.pyplot as plt
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

  def plot_cohort_frequencies(self, word_groups):
    """Plot frequencies of word groups by era.

    Args:
      word_groups: array of arrays comprising word groups.

    Usage:
      plotter.plot_cohort_frequencies([
        ['war', 'shortage', 'casualties'],
        ['thou', 'didst', 'hast']
      ])
    """
    for word_group in word_groups:
      # Sum frequencies of all words in the group.
      for i in range(len(word_group)):
        if i == 0:
          sum_frequencies = self.counts[word_group[0]]
        else:
          sum_frequencies += self.counts[word_group[i]]

      plt.plot(sum_frequencies,
        label='Cohort for \'{}\''.format(word_group[0]))

    plt.title('Word frequencies')
    plt.ylabel('Normalized word frequency')
    plt.xlabel('Era')
    plt.legend()
    plt.show()

def main():
  logging.basicConfig(format='[%(name)s %(asctime)s]\t%(msg)s',
    stream=sys.stderr, level=logging.DEBUG)
  plotter = Plotter()
  plotter.load_counts('ab_counts.pickle')
  # plotter.plot_cohort_frequencies([
  #   ['war', 'shortage', 'casualties'],
  #   ['thou', 'didst', 'hast']
  # ])
  plotter.plot_cohort_frequencies([
    ['he', 'him', 'his', 'man', 'men'],
    ['she', 'her', 'hers', 'woman', 'women']
  ])

if __name__ == '__main__':
  main()
