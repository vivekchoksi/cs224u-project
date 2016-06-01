#!/usr/bin/python

"""This module implements a featurizer tool that extracts stylistic features
from a corpus.
"""

import os
import logging
import sys
import pdb
import cPickle as pickle
from collections import defaultdict, Counter
import nltk
import nltk.data, nltk.tag
from nltk.tag.perceptron import PerceptronTagger

import util

CORPUS_DIR = os.path.join(os.path.dirname(__file__), '../data/20thCenturyCorpus/')
FEATURES_DIR = os.path.join(os.path.dirname(__file__), '../data/features/')

class FeaturizerManager(object):
  def __init__(self, featurizers_list=None):
    if featurizers_list is None:
      featurizers_list = [
        # MalePronounFeaturizer(),
        # FemalePronounFeaturizer(),
        # MaleFemalePronounRatioFeaturizer(),
        # TypeTokenRatioFeaturizer(),
        # WordCountFeaturizer(),
        # VocabSizeFeaturizer(),
        PartOfSpeechFeaturizer()
      ]
    self.featurizers_list = featurizers_list
    self.tagger = PerceptronTagger()

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
          lower_split_tokens = util.tokenize_words(line, stem=False)
          kwargs = {
            'lower_split': lower_split_tokens,
            'tags': self._get_pos_counter(lower_split_tokens),
          }
          for featurizer in self.featurizers_list:
            featurizer.process(line, book_id, **kwargs)

    for featurizer in self.featurizers_list:
      featurizer.dump()

  def get_characteristic(self, feature_name, book_id):
    return self.featurizers_map[feature_name].get_value(book_id)

  def get_part_of_speech_count(self, pos_list, book_id):
    return self.featurizers_map['pos_all'].get_value(pos_list, book_id)

  def get_book_ids(self):
    return [util.filename_to_book_id(filename) for filename in self.get_corpus_filenames()]

  def get_corpus_filenames(self):
    return os.listdir(CORPUS_DIR)

  def get_corpus_filepath(self, filename):
    return os.path.join(CORPUS_DIR, filename)

  def _get_pos_counter(self, tokens):
    tags = nltk.tag._pos_tag(tokens, None, self.tagger)
    return Counter([pos for word, pos in tags])

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
  MALE_PRONOUNS = set(['him', 'his', 'he', 'himself'])

  def __init__(self):
    super(MalePronounFeaturizer, self).__init__()
    self.feature_value = defaultdict(lambda: 0)

  def get_feature_name(self):
    return 'male_pronouns'

  def process(self, line, book_id, **kwargs):
    if 'lower_split' in kwargs.keys():
      for word in kwargs['lower_split']:
        if word in self.MALE_PRONOUNS:
          self.feature_value[book_id] += 1

class FemalePronounFeaturizer(AbstractFeaturizer):
  FEMALE_PRONOUNS = set(['she', 'her', 'hers', 'herself'])

  def __init__(self):
    super(FemalePronounFeaturizer, self).__init__()
    self.feature_value = defaultdict(lambda: 0)

  def get_feature_name(self):
    return 'female_pronouns'

  def process(self, line, book_id, **kwargs):
    if 'lower_split' in kwargs.keys():
      for word in kwargs['lower_split']:
        if word in self.FEMALE_PRONOUNS:
          self.feature_value[book_id] += 1

class FirstPersonFeaturizer(AbstractFeaturizer):
  FIRST_PERSON_PRONOUNS = set(['i', 'me', 'myself', 'my', 'mine',
    'we', 'our', 'ours', 'us', 'ourself', 'ourselves'])

  def __init__(self):
    super(FirstPersonFeaturizer, self).__init__()
    self.feature_value = defaultdict(lambda: 0)

  def get_feature_name(self):
    return 'first_person'

  def process(self, line, book_id, **kwargs):
    if 'lower_split' in kwargs.keys():
      for word in kwargs['lower_split']:
        if word in self.FIRST_PERSON_PRONOUNS:
          self.feature_value[book_id] += 1

class ThirdPersonFeaturizer(AbstractFeaturizer):
  THIRD_PERSON_PRONOUNS = set(['they', 'them', 'themself', 'themselves',
    'their', 'theirs', 'he', 'she', 'it', 'her', 'hers', 'him', 'his', 'its',
    'herself', 'himself,' 'itself'])

  def __init__(self):
    super(ThirdPersonFeaturizer, self).__init__()
    self.feature_value = defaultdict(lambda: 0)

  def get_feature_name(self):
    return 'third_person'

  def process(self, line, book_id, **kwargs):
    if 'lower_split' in kwargs.keys():
      for word in kwargs['lower_split']:
        if word in self.THIRD_PERSON_PRONOUNS:
          self.feature_value[book_id] += 1

class VocabSizeFeaturizer(AbstractFeaturizer):

  def __init__(self):
    super(VocabSizeFeaturizer, self).__init__()
    self.feature_value = defaultdict(lambda: 0)
    self.word_set = defaultdict(lambda: set())

  def get_feature_name(self):
    return 'vocab_size'

  def process(self, line, book_id, **kwargs):
    if 'lower_split' in kwargs.keys():
      for word in kwargs['lower_split']:
        if word not in self.word_set[book_id]:
          self.feature_value[book_id] += 1
          self.word_set[book_id].add(word)

class WordCountFeaturizer(AbstractFeaturizer):

  def __init__(self):
    super(WordCountFeaturizer, self).__init__()
    self.feature_value = defaultdict(lambda: 0)

  def get_feature_name(self):
    return 'word_count'

  def process(self, line, book_id, **kwargs):
    if 'lower_split' in kwargs.keys():
      for word in kwargs['lower_split']:
        self.feature_value[book_id] += 1

class TypeTokenRatioFeaturizer(AbstractFeaturizer):

  def __init__(self):
    super(TypeTokenRatioFeaturizer, self).__init__()
    self.type_featurizer = VocabSizeFeaturizer()
    self.token_featurizer = WordCountFeaturizer()

  def get_feature_name(self):
    return 'type_token_ratio'

  def get_value(self, book_id):
    types = self.type_featurizer.get_value(book_id)
    tokens = self.token_featurizer.get_value(book_id)
    return float(types) / tokens

  def dump(self):
    pass

class PartOfSpeechFeaturizer(AbstractFeaturizer):
  # Enumeration of parts of speech:
  # https://cs.nyu.edu/grishman/jet/guide/PennPOS.html

  def __init__(self):
    super(PartOfSpeechFeaturizer, self).__init__()
    self.feature_value = defaultdict(lambda: Counter())

  def get_feature_name(self):
    return 'pos_all'

  def process(self, line, book_id, **kwargs):
    if 'tags' in kwargs.keys():
      self.feature_value[book_id] += kwargs['tags']

  def get_value(self, pos_list, book_id):
    """Get the total number of occurrences of parts of speech in the input
    part of speech list for the book with the given book id.
    """
    if self.feature_dict is None:
      self.feature_dict = util.pickle_load(self._get_pickle_filename())

    sum_count = 0
    for pos in pos_list:
      sum_count += self.feature_dict[book_id][pos]
    return sum_count


def run_featurizer():
  featurizer_manager = FeaturizerManager()
  featurizer_manager.run_featurizers()
  # pdb.set_trace()
  # featurizer_manager.get_characteristic('male_pronouns', '21')

def main():
  logging.basicConfig(format='[%(name)s %(asctime)s]\t%(msg)s',
    stream=sys.stderr, level=logging.DEBUG)
  run_featurizer()

if __name__ == '__main__':
  main()
