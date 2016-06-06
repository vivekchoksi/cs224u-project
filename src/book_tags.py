#!/usr/bin/python

"""
For the Twentieth Century Corpus,
creates a dict of {book_id : {k1:v1, k2:v2,...}}, 
mapping book_id to a dict of key:value pairs
"""

import os
import string
import cPickle as pickle
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
		wr.writerow(sh.row_values(rownum))

	csv_file.close()

def get_filepath(book_id):
	corpus_dir = os.path.join(DATA_DIR, CORPUS_SUBDIR)
	for filename in os.listdir(corpus_dir):
		filepath = os.path.join(corpus_dir, filename)

		match = re.match(r"([0-9]+)([a-z]+)([0-9]+)", filename.lower().replace(" ",""), re.I)
		if match:
			items = match.groups()
			curr_book_id = items[0]

			if curr_book_id == book_id:
				return filepath

	print "%s -> filepath match not found" % (book_id)


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
			if key[0] == 'heinlein':
				if title == 'the door into summer':
					values['filepath'] = get_filepath("945")
					book_metadata["945"] = values
					matches += 1
					book_id = "945"
					if book_id in id_set:
						print "Dup id: ", book_id
					else:
						id_set.add(book_id)
					continue

				if title == 'double star':
					values['filepath'] = get_filepath("942")
					book_metadata["942"] = values
					matches += 1
					book_id = "942"
					if book_id in id_set:
						print "Dup id: ", book_id
					else:
						id_set.add(book_id)
					continue

			# search for matching (last name, year) key
			for filename in os.listdir(corpus_dir):
				curr_key = None

				match = re.match(r"([0-9]+)([a-z]+)([0-9]+)", filename.lower().replace(" ",""), re.I)
				if match:
					items = match.groups()
					curr_key = (items[1], int(items[2]))
					book_id = items[0]

				if curr_key == key:
					values['filepath'] = get_filepath(book_id)
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
							match = re.match(r"([0-9]+)([a-z]+)([0-9]+)", filename.lower().replace(" ", ""), re.I)
							if match:
								items = match.groups()
								curr_key = (items[1], int(items[2]))
								book_id = items[0]
								matches += 1
								found_match = True
								values['filepath'] = get_filepath(book_id)
								# if book_id in id_set:
								# 	print "Dup id from line matching: ", book_id, title
								# else:
								# 	id_set.add(book_id)

								book_metadata[book_id] = values
								continue
					if found_match:
						break

			if not found_match: 
				print "Couldn't find match for: ",  title

	#found_filenames = set(book_metadata[book_id]['filepath'] for book_id in book_metadata)
	#print all_filenames - found_filenames

	#print count, matches




	return book_metadata


#xlsx_to_csv(os.path.join(DATA_DIR, 'book_metadata.xlsx'), os.path.join(DATA_DIR, 'book_metadata.csv'))
book_metadata = create_book_tags(os.path.join(DATA_DIR, 'book_metadata.csv'))
# print len(book_metadata)

# for book_id in book_metadata:
# 	print book_id
# 	if 'filepath' not in book_metadata[book_id]:
# 		print "key \"filepath\" not found"
# 	print '\n'

# util.pickle_dump(book_metadata, 'data/pickle/book_metadata.pickle')



