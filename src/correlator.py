#!/usr/bin/python

"""This module implements a Correlator tool as described by The Stanford
Literary Lab.

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
from collections import Counter
from scipy.stats.stats import pearsonr

import util

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/')


class BookReader():
  """
  Read data from the corpus by book.
  """

  def __init__(self):
    pass

  def _get_filenames(self, era):
    # Hard-coded example.
    if era == 0:
      filenames = ['ebooks-unzipped/1895/12190-8.txt']
    elif era == 1:
      filenames = ['ebooks-unzipped/1923/1156.txt']
    else:
      raise ValueError('Invalid era.')

    return [os.path.join(DATA_DIR, f) for f in filenames]

  def read_book(self):
    '''
    By default, yield empty data
    '''
    for file in self._get_filenames():
      yield(file, None)



class WordCountsBookReader(BookReader):
  """
  Yields Counter of words for each book
  """

  START_YEAR = 1895
  END_YEAR = 1923

  def __init__(self, era_size=5):
    self.era_size=era_size
    self.num_eras = int(np.ceil(\
      float(self.END_YEAR - self.START_YEAR + 1) / era_size))

  def read_book(self):
    already_processed = set()

    for year, file in self._get_filenames():
      print file
      title = util.fn_to_title(file)
      print title

      if title not in already_processed:
        already_processed.add(title)
        c = Counter()
        with open(file, 'r') as infile:
          #print file
          for line in infile:
            # Make lowercase, remove punctuation, and split into words.
            c.update(util.tokenize_words(line))
            #c.update(line.split()
        yield (title, (year, c))

  def _get_filenames(self):

    start_era = self.START_YEAR
    end_era = self.END_YEAR + 1

    filenames = []
    corpus_dir = os.path.join(DATA_DIR, 'ebooks')
    for year in range(start_era, end_era):
      year_dir = os.path.join(corpus_dir, str(year))
      for filename in os.listdir(year_dir):
        if not filename.startswith('.'):
          filenames.append((year, os.path.join(year_dir, filename)))

    return filenames

class TwentiethCenturyWordCountsBookReader(WordCountsBookReader):
  """
  Yields Counter of words for each book
  """

  START_YEAR = 1900
  END_YEAR = 1999
  CORPUS_SUBDIR = '20thCenturyCorpus'

  def __init__(self, era_size=5):
    self.era_size=era_size
    self.num_eras = int(np.ceil(\
      float(self.END_YEAR - self.START_YEAR + 1) / era_size))

  def read_book(self):
    already_processed = set()

    for year, file in self._get_filenames():
      print file
      title = file.split('/')[-1]
      print title

      if title not in already_processed:
        already_processed.add(title)
        c = Counter()
        with open(file, 'r') as infile:
          #print file
          for line in infile:
            # Make lowercase, remove punctuation, and split into words.
            c.update(util.tokenize_words(line))
            #c.update(line.split()
        yield (title, (year, c))

  def _get_filenames(self):
    tcr = TwentiethCenturyReader()
    filenames = []
    for era in range(tcr.get_num_eras()):
      for filename in tcr._get_filenames(era):
        filenames.append((tcr._get_book_year(filename), filename))
    return filenames


class BookDataProcessor():
  """
  Given a BookReader to yield (book, data) pairs, 
  processes the pairs and stores the resulting data (for use by, say, some Plotter)
  """

  def __init__(self, book_reader):
    self.book_reader = book_reader

  def process_book_data(self):
    for book_title, data in self.book_reader.read_book():
      # process the book and data
      continue



class WordCountsBookDataProcessor(BookDataProcessor):

  def preprocess_word_counts(self, pickle_dump_file=None):
    """
    For each book, store a tuple of (year, word counter, total words)
    """

    self.title_to_data = {}

    for title, data in self.book_reader.read_book():
      year, counter = data
      self.title_to_data[title] = (year, counter, sum(counter.values()))

    if pickle_dump_file is not None:
      util.pickle_dump(self.title_to_data, pickle_dump_file)


  def load_word_counts(self, pickle_dump_file):
    self.title_to_data = util.pickle_load(pickle_dump_file)


  def get_data_all_books(self, words):
    """
    given ist of words,
    return three lists, in parallel order:
      titles (labels)
      year (x axis)
      normalized word count (y axis)
    """

    titles = self.title_to_data.keys()
    years = []
    normalized_wcs = []

    for title in titles:
      data = get_data_single_book(title, words)
      years.append(data[0])
      normalized_wcs.append(data[1])

    return titles, years, normalized_wcs


  def get_data_single_book(self, title, words):
    """
    given book title and list of words,
    return tuple of (year, normalized word count) 
    """

    words = util.stem_words(words)

    year, counter, total_wc = self.title_to_data[title]

    raw_wc = sum([counter[word] for word in words])
    normalized_wc = raw_wc / float(total_wc)

    return (year, normalized_wc)






class Reader(object):
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
          try:
            # Make lowercase, remove punctuation, and split into words.
            for word in util.tokenize_words(line):
              yield word
          except:
            logging.error(
              'Unable to read a line in the file: {}'.format(infile))
            logging.error(line)
            logging.error('Skipping...')

  def get_num_eras(self):
    # Hard-coded example.
    return 2

  def _get_filenames(self, era):
    # Hard-coded example.
    if era == 0:
      filenames = ['ebooks/1895/Beside the Bonnie Brier Bush, 7179-8.txt']
    elif era == 1:
      filenames = ['ebooks/1923/Babbitt, 1156.txt']
    else:
      raise ValueError('Invalid era.')

    return [os.path.join(DATA_DIR, f) for f in filenames]


class TccListReader(Reader):
  """
  Divides Twentieth Century Corpus into eras by list
  """

  I_TO_LIST_NAME = [
    'ml_editors_rank',
    'ml_readers_rank',
    'pub_prog_rank',
    'exp_rank',
    'bestsellers_rank',
  ]

  def __init__(self):
    book_metadata = util.pickle_load(os.path.join(DATA_DIR, "pickle/book_metadata.pickle"))
    book_lists = [ [] for i in range(5) ] #5 lists; hardcoded

    for book_id in book_metadata:
      metadata = book_metadata[book_id]

      for i in range(5):
        # if book has (non zero) rank on list
        if metadata[self.I_TO_LIST_NAME[i]] != 0:
          book_lists[i].append(metadata['filepath'])

    self.book_lists = book_lists

  def get_num_eras(self):
    return len(self.book_lists)

  def _get_filenames(self, era):
    return self.book_lists[era]


class AbstractCorpusReader(Reader):
  """Abstract class to read data from a corpus.

  Contains logic for splitting year ranges into eras.
  """
  START_YEAR = None
  END_YEAR = None
  CORPUS_SUBDIR = None

  def __init__(self, era_size=5):
    """
    Args:
      era_size (int): the number of years that constitute an era.
    """
    self.era_size = era_size

    # Count "incomplete eras" (e.g. round up if the time period contains
    # some number of eras plus a fraction of an era).
    self.num_eras = int(np.ceil(
      float(self.END_YEAR - self.START_YEAR + 1) / era_size))

  def get_num_eras(self):
    return self.num_eras

  def _get_years_for_era(self, era):
    if era < 0 or era >= self.num_eras:
      raise ValueError('Invalid era.')

    start_era = self.START_YEAR + era * self.era_size
    end_era = min(self.START_YEAR + (era + 1) * self.era_size,
      self.END_YEAR + 1)

    return range(start_era, end_era)

  def _get_filenames(self, era):
    pass

class AmericanBestsellersReader(AbstractCorpusReader):
  """Read data from American Bestsellers list as provided by Project Gutenberg.
  https://www.gutenberg.org/wiki/Bestsellers,_American,_1895-1923_(Bookshelf)
  """
  START_YEAR = 1895
  END_YEAR = 1923
  CORPUS_SUBDIR = 'ebooks'

  def __init__(self, era_size=5):
    super(AmericanBestsellersReader, self).__init__(era_size)

  def _get_filenames(self, era):
    filenames = []
    corpus_dir = os.path.join(DATA_DIR, self.CORPUS_SUBDIR)
    for year in self._get_years_for_era(era):
      year_dir = os.path.join(corpus_dir, str(year))
      for filename in os.listdir(year_dir):
        filenames.append(os.path.join(year_dir, filename))

    return filenames


class TwentiethCenturyReader(AbstractCorpusReader):
  """Read data from the 20th Century American corpus as provided by the
  Stanford Literary Lab and described in their eighth pamphlet.

  http://litlab.stanford.edu/LiteraryLabPamphlet8.pdf
  """
  # The true start and end years.
  TRUE_START_YEAR = 1881
  TRUE_END_YEAR = 2011

  # The start and end years to study.
  START_YEAR = 1900
  END_YEAR = 1999
  CORPUS_SUBDIR = '20thCenturyCorpus'

  # Need to rename files that are not annotated with the book's publication
  # year.
  FILES_TO_RENAME = {
    '1133JoyceNA.txt': '1133Joyce1941.txt',
    '1171KeneallyNA.txt': '1171Keneally1982.txt',
    '1201KeseyNA.txt': '1201Kesey1962.txt',
    '1271LarssonNA.txt': '1271Larsson2007.txt',
    '1281LawrenceNA.txt': '1281Lawrence1928.txt',
    '156OBrian39.txt': '156OBrien1951.txt',
    '1571OBrien53.txt': '1571OConnor1953.txt',
    '1651PynchonNA.txt': '1651Pynchon1973.txt',
    '1711RipleyNA.txt': '1711Ripley1991.txt',
    '1751RothNA.txt': '1751Roth1969.txt',
    '1761RushdieNA.txt': '1761Rushdie1981.txt',
    '1911StyronNA.txt': '1911Styron1979.txt',
    '1951TherouxNA.txt': '1951Theroux1981.txt',
    '2021VonnegutNA.txt': '2021Vonnegut1963.txt',
    '2071WarrenNA.txt': '2071Warren1946.txt',
    '2141WhiteNA.txt': '2141White1952.txt',
    '221BennettNA.txt': '221Bennett1908.txt',
    '261BrinkleyNA.txt': '261Brinkley1956.txt',
    '31AdamsNA.txt': '31Adams1979.txt',
    '351CardNA.txt': '351Card1985.txt',
    '612du MaurierNA.txt': '612du Maurier1946.txt',
    '761FordNA.txt': '761Ford1925.txt',
    '781FowlesNA.txt': '781Fowles1969.txt',
    '802GassNA.txt': '802Gass1966.txt',
    '91AtwoodNA.txt': '91Atwood1985.txt',
  }

  def __init__(self, era_size=10):
    super(TwentiethCenturyReader, self).__init__(era_size)

  def _get_filenames(self, era):
    # print 'Getting filenames for era', era
    filenames = []
    corpus_dir = os.path.join(DATA_DIR, self.CORPUS_SUBDIR)

    # Get books in a particular year by looping through all filenames.
    # Not efficient, but also not worth optimizing.
    for year in self._get_years_for_era(era):
      for filename in os.listdir(corpus_dir):
        book_year = self._get_book_year(filename)
        if year == book_year:
          filenames.append(os.path.join(corpus_dir, filename))

    return filenames

  def _get_book_year(self, filename):
    """Return the year from the filename of a txt file in the 20th Century
    Corpus. The year occupies the last four characters of the file name.
    """
    if filename in self.FILES_TO_RENAME:
      filename = self._rename_file(filename)

    year = int(filename[-8:-4])

    assert(year >= self.TRUE_START_YEAR and year <= self.TRUE_END_YEAR)
    return year

  def _rename_file(self, filename):
    """Rename the file to include the book publication year, which some file
    names are missing.
    """
    corpus_dir = os.path.join(DATA_DIR, self.CORPUS_SUBDIR)
    old_name = os.path.join(corpus_dir, filename)
    new_name = os.path.join(corpus_dir, self.FILES_TO_RENAME[filename])
    os.rename(old_name, new_name)
    logging.info('Renamed file \'{}\' to \'{}\''.format(old_name, new_name))
    return new_name


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

  def get_most_fluctuating_words(self, fluctuation_fn=None, k=20):
    """Reports the words for which the frequencies fluctuated most, given
    some input function to measure fluctuation.

    Args:
      fluctuation_fn: a function to report a fluctuation score given an array
        representing frequency counts of a word.
      k (int): the number of most highly fluctuating words to report.
    """
    # Default fluctuation function.
    if fluctuation_fn is None:
      fluctuation_fn = self.elementwise_fluctuation

    fluctuations = {}
    for word in self.correlations:
      # Compute a single fluctuation score for the word.
      fluctuations[word] = fluctuation_fn(self.counts[word])

    # Sort to get the most fluctuating cohorts
    fluctuations = sorted(
      fluctuations.items(), key=operator.itemgetter(1), reverse=True)

    # Print the top several most fluctuating cohorts.
    print 'Words that see the most fluctuation in frequency:'
    for word, _ in fluctuations[:k]:
      cohort_words = [w for w, _ in self.get_cohort(word)]
      # print '...', word, ':', self.counts[word]
      print '...', word

    return fluctuations[:k]

  def get_most_fluctuating_cohorts(
    self, fluctuation_fn=None, k=20):
    """Reports the cohorts for which the frequencies fluctuated most, given
    some input function to measure fluctuation.

    Args:
      fluctuation_fn: a function to report a fluctuation score given an array
        representing frequency counts of a semantic cohort.
      k (int): the number of most highly fluctuating cohorts to report.
    """
    logging.info('Computing highest fluctuating cohorts...')

    # Default fluctuation function.
    if fluctuation_fn is None:
      fluctuation_fn = self.elementwise_fluctuation

    fluctuations = {}
    for word in self.correlations:
      # Get sum of frequencies for all words in cohort.
      sum_counts = np.copy(self.counts[word])
      for other_word, _ in self.get_cohort(word):
        if other_word is not word:
          sum_counts += self.counts[other_word]

      # Compute a single fluctuation score for the word's cohort.
      fluctuations[word] = fluctuation_fn(sum_counts)

    # Sort to get the most fluctuating cohorts
    fluctuations = sorted(
      fluctuations.items(), key=operator.itemgetter(1), reverse=True)

    # Print the top several most fluctuating cohorts.
    print 'Cohorts that see the most fluctuation in frequency:'
    for word, _ in fluctuations[:k]:
      cohort_words = [w for w, _ in self.get_cohort(word)]
      print '...', word, ':', cohort_words

    return fluctuations[:k]

  def std_dev_times_freq_fluctuation(self, array):
    return array.std() * np.sum(array)

  def std_dev_fluctuation(self, array):
    return array.std()

  def elementwise_fluctuation(self, array):
    fluctuation = 0.0
    for i in range(array.size - 1):
      fluctuation += np.abs(array[i+1] - array[i])
    return fluctuation

  def sum_only_upward_fluctuation(self, array):
    fluctuation = 0.0
    for i in range(array.size - 1):
      increase = array[i+1] - array[i]
      fluctuation += increase / array[i] if increase > 0 else 0
    return fluctuation

  def sum_only_downward_fluctuation(self, array):
    fluctuation = 0.0
    for i in range(array.size - 1):
      decrease = array[i] - array[i-1]
      fluctuation += decrease / array[i] if decrease > 0 else 0
    return fluctuation

  def upward_fluctuation(self, array):
    return array[-1] - array[0]

  def downward_fluctuation(self, array):
    return array[0] - array[-1]

  def upward_fluctuation_between_halves(self, array):
    first_half = 0.0
    second_half = 0.0
    for i in range(array.size):
      if i + 1 <= array.size / 2:
        first_half += array[i]
      else:
        second_half += array[i]
    return second_half - first_half

  def downward_fluctuation_between_halves(self, array):
    return -self.upward_fluctuation_between_halves(array)

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

    # NOTE(vivekchoksi): This is very slow; would take ~15 hours to run on my
    # laptop. Might be faster if this operation were vectorized; that is, by
    # computing Pearson coefficients using matrices instead of a loop.
    for w1 in self.counts:
      correlations[w1] = self._correlate_word(w1, k)

      counter += 1
      if counter % 10 == 0:
        logging.info('\tFinished correlating {} of {} words...'.format(
          counter, len(self.counts)))

      # if counter == 150:
      #   break

    self.correlations = dict(correlations)

  def _correlate_word(self, word, k):
    """Return the k words that correlate most highly with the input word.

    Similarity is calculated using the Pearson correlation coefficient.
    """
    # Correlations for `word`. E.g. correlations[word2] = Pearson coefficient
    correlations = {}

    for word2 in self.counts:
      if word is not word2:
        correlations[word2] = pearsonr(
          self.counts[word], self.counts[word2])[0]

    # To save space, only store top k correlations. That is, for each w1,
    # only store k w2. The trade-off is that there is redundant computation.
    return sorted(
      correlations.items(), key=operator.itemgetter(1), reverse=True)[:k]


def run_correlator():
  correlator = Correlator()
  # correlator.preprocess_counts(TwentiethCenturyReader(), 'data/pickle/tcc_counts_1900-1999.pickle')
  # correlator.preprocess_correlations(k=20, pickle_dump_file='data/pickle/tcc_correlations_1900-1999.pickle')
  correlator.load_counts('data/pickle/tcc_counts_1900-1999.pickle')
  correlator.load_correlations('data/pickle/tcc_correlations_1900-1999.pickle')
  # correlator.preprocess_counts(TccListReader(), 'data/pickle/tcc_counts_by_list.pickle')
  # correlator.preprocess_correlations(k=20, pickle_dump_file='data/pickle/tcc_correlations_by_list.pickle')
  # correlator.load_counts('data/pickle/tcc_counts_by_list.pickle')
  # correlator.load_correlations('data/pickle/tcc_correlations_by_list.pickle')


  correlator.get_most_fluctuating_words(lambda arr: (np.max(arr) - np.min(arr)) / np.min(arr))

  print 'Most upward trending words:'
  correlator.get_most_fluctuating_words(correlator.sum_only_upward_fluctuation)
  print 'Most downward trending words:'
  correlator.get_most_fluctuating_words(correlator.sum_only_downward_fluctuation)


  # correlator.get_most_fluctuating_words(correlator.upward_fluctuation_between_halves)
  # correlator.get_most_fluctuating_words(correlator.downward_fluctuation_between_halves)

 
def run_wc_by_book():
  wcp = WordCountsBookDataProcessor(TwentiethCenturyWordCountsBookReader())
  wcp.preprocess_word_counts("tcc_wc_by_book.pickle")
  #wcp.load_word_counts("wc_by_book.pickle")
  # print wcp.get_data_single_book("Beside the Bonnie Brier Bush", ['brier'])



def main():
  logging.basicConfig(format='[%(name)s %(asctime)s]\t%(msg)s',
    stream=sys.stderr, level=logging.DEBUG)
  # run_correlator()
  run_wc_by_book()
  # test_tcc_list_reader()

if __name__ == '__main__':
  main()
