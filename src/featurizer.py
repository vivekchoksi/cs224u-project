#!/usr/bin/python

"""This module implements a featurizer tool that extracts stylistic features
from a corpus.
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
from collections import Counter

import util

CORPUS_DIR = os.path.join(os.path.dirname(__file__), '../data/20thCenturyCorpus/')
FEATURES_DIR = os.path.join(os.path.dirname(__file__), '../data/features/')


class FeaturizerManager(object):
  def __init__(self, featurizers_list=None):
    if featurizers_list is None:
      featurizers_list = [
        MalePronounFeaturizer(),
        FemalePronounFeaturizer(),
        MaleFemalePronounRatioFeaturizer()
      ]
    self.featurizers_list = featurizers_list

    self.featurizers_map = {}
    for f in featurizers_list:
      self.featurizers_map[f.get_feature_name()] = f

  def run_featurizers(self):
    filenames = self.get_corpus_filenames()
    for i, filename in enumerate(filenames):
      logging.info('Reading book {} / {}: {}'.format(i+1, len(filenames), filename))
      book_id = util.filename_to_book_id(filename)
      filepath = self.get_corpus_filepath(filename)
      with open(filepath, 'r') as file:
        for line in file:
          kwargs = {}
          for featurizer in self.featurizers_list:
            featurizer.process(line, book_id, **kwargs)

    for featurizer in self.featurizers_list:
      featurizer.dump()

  def get_characteristic(self, feature_name, book_id):
    return self.featurizers_map[feature_name].get_value(book_id)

  def get_book_ids(self):
    return [util.filename_to_book_id(filename) for filename in self.get_corpus_filenames()]

  def get_corpus_filenames(self):
    return os.listdir(CORPUS_DIR)

  def get_corpus_filepath(self, filename):
    return os.path.join(CORPUS_DIR, filename)


class AbstractFeaturizer(object):

  def __init__(self):
    self.features_dir = FEATURES_DIR
    self.feature_value = None
    self.feature_dict = None

  def get_feature_name(self):
    return None

  def process(self, line, book_id, **kwargs):
    pass

  def dump(self):
    util.pickle_dump(dict(self.feature_value), self._get_pickle_filename())

  def get_value(self, book_id):
    if self.feature_dict is None:
      self.feature_dict = util.pickle_load(self._get_pickle_filename())
    return self.feature_dict[book_id]

  def _get_pickle_filename(self):
    return self.features_dir + self.get_feature_name() + '.pickle'

class MaleFemalePronounRatioFeaturizer(AbstractFeaturizer):

  def __init__(self):
    super(MaleFemalePronounRatioFeaturizer, self).__init__()
    self.male_pronoun_featurizer = MalePronounFeaturizer()
    self.female_pronoun_featurizer = FemalePronounFeaturizer()

  def get_feature_name(self):
    return 'male_female_pronoun_ratio'

  def get_value(self, book_id):
    male_pronouns = self.male_pronoun_featurizer.get_value(book_id)
    female_pronouns = self.female_pronoun_featurizer.get_value(book_id)
    return float(male_pronouns) / female_pronouns

  def dump(self):
    pass

class MalePronounFeaturizer(AbstractFeaturizer):
  MALE_PRONOUNS = set(['him', 'his', 'he'])

  def __init__(self):
    super(MalePronounFeaturizer, self).__init__()
    self.feature_value = defaultdict(lambda: 0)

  def get_feature_name(self):
    return 'male_pronouns'

  def process(self, line, book_id, **kwargs):
    for word in line.split():
      if word in self.MALE_PRONOUNS:
        self.feature_value[book_id] += 1

class FemalePronounFeaturizer(AbstractFeaturizer):
  FEMALE_PRONOUNS = set(['she', 'her', 'hers'])

  def __init__(self):
    super(FemalePronounFeaturizer, self).__init__()
    self.feature_value = defaultdict(lambda: 0)

  def get_feature_name(self):
    return 'female_pronouns'

  def process(self, line, book_id, **kwargs):
    for word in line.split():
      if word in self.FEMALE_PRONOUNS:
        self.feature_value[book_id] += 1

def run_featurizer():
  featurizer_manager = FeaturizerManager()
  featurizer_manager.run_featurizers()
  pdb.set_trace()
  featurizer_manager.get_characteristic('male_pronouns', '21')


def main():
  logging.basicConfig(format='[%(name)s %(asctime)s]\t%(msg)s',
    stream=sys.stderr, level=logging.DEBUG)
  run_featurizer()

if __name__ == '__main__':
  main()
