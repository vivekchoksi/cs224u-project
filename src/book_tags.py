#!/usr/bin/python

"""
For the Twentieth Century Corpus,
creates a dict of {book_id : {k1:v1, k2:v2,...}}, 
mapping book_id to a dict of key:value pairs
"""

import os
# import operator
# import numpy as np
# import logging
# import sys
import string
# import pdb
import cPickle as pickle
# from collections import defaultdict
# from collections import Counter
# from scipy.stats.stats import pearsonr
import xlrd
import csv
import editdistance
import re

import util

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/')
CORPUS_SUBDIR = '20thCenturyCorpus'

def xlsx_to_csv(xlsx_filepath, csv_filepath):
	"""
	Taken from Stack Overflow
	source: http://stackoverflow.com/questions/20105118/convert-xlsx-to-csv-correctly-using-python
	"""
	wb = xlrd.open_workbook(xlsx_filepath)
	sh = wb.sheet_by_name('Sheet1')
	csv_file = open(csv_filepath, 'wb')
	wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
	for rownum in xrange(sh.nrows):

		#print "rownum: ", rownum, sh.row_values(rownum)
		wr.writerow(sh.row_values(rownum))

	csv_file.close()

def create_book_tags(filepath):

	corpus_dir = os.path.join(DATA_DIR, CORPUS_SUBDIR)

	book_metadata = {}
	count = 0
	matches = 0

	with open(filepath, 'rb') as f:
		reader = csv.reader(f)
		# for each book
		for row in reader:
			found_match = False
			book_id = None
			count += 1
			if count == 1:
				continue
			
			metadata = row	
			values = {
				'title': metadata[0],
				'author': metadata[1],
				'year': int(float(metadata[2])),
				'ml_editors_rank': int(float(metadata[3])),
				'ml_readers_rank': int(float(metadata[4])),
				'pub_prog_rank': int(float(metadata[5])),
				'exp_rank': int(float(metadata[6])),
				'bestsellers_rank': int(float(metadata[7])),
				'gender': metadata[8],
				'natl': metadata[9],
				'num_lists': metadata[10],
			}

			title = metadata[0]
			last_name = metadata[1].split()[-1].lower()
			year = int(float(metadata[2]))

			key = (last_name, year)

			# hard coded key collision cases
			if key == ('heinlein', 1957):
				if title == 'citizen of the galaxy':
					book_metadata["941"] = values
					matches += 1
					continue

				if title == 'the door into summer':
					book_metadata["945"] = values
					matches += 1
					continue

			if key == ('lint', 1984):
				book_metadata["503"] = values
				matches += 1
				continue

			if key == ('lint', 1988):
				book_metadata["501"] = values
				matches += 1
				continue



			# search for matching (last name, year) key
			for filename in os.listdir(corpus_dir):
				curr_key = None

				match = re.match(r"([0-9]+)([a-z]+)([0-9]+)", filename.lower(), re.I)
				if match:
					items = match.groups()
					curr_key = (items[1], int(items[2]))
					book_id = items[0]

				if curr_key == key:
					book_metadata[book_id] = values
					found_match = True
					matches += 1
					break

			if found_match:
				continue

			# search for title in first 5 lines of each text file
			for filename in os.listdir(corpus_dir):
				text_path = os.path.join(corpus_dir, filename)
				with open(text_path) as text:
					for i in range(5):
						curr_title = text.readline().lower().strip()
						edit_dist = int(editdistance.eval(curr_title, title))
						if edit_dist < int(float(len(title)) / 10) or edit_dist == 0:
							match = re.match(r"([0-9]+)([a-z]+)([0-9]+)", filename.lower(), re.I)
							if match:
								items = match.groups()
								curr_key = (items[1], int(items[2]))
								book_id = items[0]
							matches += 1
							found_match = True
							book_metadata[book_id] = values
							continue
					if found_match:
						break

			if not found_match: 
				print "Couldn't find match for: ",  title

	#print count, matches
	return book_metadata


# xlsx_to_csv(os.path.join(DATA_DIR, 'book_metadata.xlsx'), os.path.join(DATA_DIR, 'book_metadata.csv'))
book_metadata = create_book_tags(os.path.join(DATA_DIR, 'book_metadata.csv'))
# for book_id in book_metadata:
# 	print book_id
# 	print book_metadata[book_id]['title']
# 	print book_metadata[book_id]['author']
# 	print "\n"

util.pickle_dump(book_metadata, 'data/pickle/book_metadata.pickle')



