'''
scrapes Project Gutenberg for its American bestsellers 1895-1923
list of books
'''

from lxml import html, etree
from bs4 import BeautifulSoup
import cPickle as pickle
import urllib
import re
from urlparse import urlparse
import sys
#import unicodedata
#import requests

import os
import zipfile
import gzip
import datetime
import codecs
import shutil
#import glob


class IdReader():
	'''
	Superclass

	Any IdReader that yields (header_text, book_id) may be used with Scraper.
	'''
	def yield_id(self):
		yield ("some_header_text", 14571)


class BookshelfIdReader(IdReader):
	'''
	IdReader for a Gutenberg "Bookshelf" page.
	'''
	def __init__(self, url):
		self.url = url

	def yield_id(self):
		'''
		Based on the url of a Gutenberg Bookshelf page given at initialization,
		yield tuples of (header_text, book_id)
		'''

		# get soup of bookshelf page from url

		#r = urllib.urlopen(self.url).read()
		#soup = BeautifulSoup(r, "lxml")

		# temporary for opening local HTML file
		soup = BeautifulSoup(open(self.url), "lxml")

		# find all elements that are headers
		headers = soup.find_all(["h2"])

		for header in headers:
			#used to create sub-directory for this group of books
			header_text = header.get_text()
	
			for link in header.find_all_next(["h2","li"]):

				# if we find the next header, break
				if link.name == "h2":
					break

				# check if link has 'a'
				if link.a and link.a['href']:
					# check if url is that of a Gutenberg book
					parsed_uri = urlparse(str(link.a['href']))	
					domain = '{uri.netloc}'.format(uri=parsed_uri)
					if domain != "www.gutenberg.org":
						continue
					
					#book_url = "https://www.gutenberg.org/files/%s/%s-h/%s-h.htm" % (book_id, book_id, book_id)
					#yield (header_text, book_url)

					book_id = re.sub('[^0-9]','',link.a['title'])

					yield (header_text, int(book_id))
			


class Scraper():
	'''
	adapted from Michiel Overtoom's Gutenberg eBook Scraper
	'''

	def __init__(self, mirror_url):
		'''
		Initialized with url of a mirror hosting Gutenberg's books.
		Requires significant preprocessing to index books' urls
		based on mirror's indexing system.
		'''
		self.mirror = mirror_url
		self.language = "English"


		def older(a, b):
			'''Return True is file 'a' is older than file 'b'.'''
			if not os.path.exists(a) or not os.path.exists(b):
				return False
			sta = os.stat(a)
			stb = os.stat(b)
			return sta <= stb


		def fetch(mirrorurl, filename, outputfilename):
			'''Fetch a file from a gutenberg mirror, if it hasn't been fetched earlier today.'''
			mustdownload = False
			if os.path.exists(outputfilename):
				st = os.stat(outputfilename)
				modified = datetime.date.fromtimestamp(st.st_mtime)
				today = datetime.date.today()
				if modified == today:
					print "%s exists, and is up-to-date. No need to download it." % outputfilename
				else:
					print "%d exists, but is out of date. Downloading..." % outputfilename
					mustdownload = True
			else:
				print "%s not found, downloading..." % outputfilename
				mustdownload = True

			if mustdownload:
				url = mirrorurl + filename
				urllib.urlretrieve(url, outputfilename)

		# Ensure directories exist.
		if not os.path.exists("indexes"):
			os.mkdir("indexes")

		if not os.path.exists("ebooks-zipped"):
			os.mkdir("ebooks-zipped")

		if not os.path.exists("ebooks-unzipped"):
			os.mkdir("ebooks-unzipped")

		# Download the book index, and unzip it.
		fetch(self.mirror, "GUTINDEX.zip", "indexes/GUTINDEX.zip")
		if not os.path.exists("indexes/GUTINDEX.ALL") or older("indexes/GUTINDEX.ALL", "indexes/GUTINDEX.zip"):
			print "Extracting GUTINDEX.ALL from GUTINDEX.zip..."
			zipfile.ZipFile("indexes/GUTINDEX.zip").extractall("indexes/")

		# Download the file index, and gunzip it.
		fetch(self.mirror, "ls-lR.gz", "indexes/ls-lR.gz")
		if not os.path.exists("indexes/ls-lR") or older("indexes/ls-lR", "indexes/ls-lR.gz"):
			print "Extracting ls-lR from ls-lR.gz..."
			inf = gzip.open("indexes/ls-lR.gz", "rb")
			outf = open("indexes/ls-lR", "wb")
			outf.write(inf.read())
			inf.close()
			outf.close()

		# Parse the file index
		if not os.path.exists("indexes/mirrordir.p") or not os.path.exists("indexes/mirrorname.p"):
			print "Parsing file index..."
			self.mirrordir = {}
			self.mirrorname = {}
			re_txt0file = re.compile(r".*? (\d+\-0\.zip)") # UTF-8 encoded (?)
			re_txt8file = re.compile(r".*? (\d+\-8\.zip)") # latin-8 encoded (?)
			re_txtfile = re.compile(r".*? (\d+\.zip)") # ascii encoded (?)
			for line in open("indexes/ls-lR"):
				if line.startswith("./"):
					line = line[2:].strip()
					if line.endswith(":"):
						line = line[:-1]
					if line.endswith("old") or "-" in line:
						continue
					lastseendir = line
					continue
				m = re_txt0file.match(line)
				if not m:
					m = re_txt8file.match(line)
				if not m:
					m = re_txtfile.match(line)
				if m:
					filename = m.groups()[0]
					if "-" in filename: # For filenames like '12104-0.zip'.
						nr, _ = filename.split("-")
					elif "." in filename: # For filenames like '32901.zip'.
						nr, _ = filename.split(".")
					else:
						print "Unexpected filename:", filename
					ebookno = int(nr)
					if not ebookno in self.mirrordir:
						self.mirrordir[ebookno] = lastseendir
						self.mirrorname[ebookno] = filename
			pickle.dump(self.mirrordir, open("indexes/mirrordir.p", "wb"))
			pickle.dump(self.mirrorname, open("indexes/mirrorname.p", "wb"))
			
		else:
			self.mirrordir = pickle.load(open("indexes/mirrordir.p", "rb"))
			self.mirrorname = pickle.load(open("indexes/mirrorname.p", "rb"))

		# Parse the GUTINDEX.ALL file and extract all language-specific titles from it.
		print "Parsing book index..."
		inpreamble = True
		ebooks = {} # number -> title
		self.ebookslanguage = {} # number -> language
		ebookno = None
		nr = 0
		langre = re.compile(r"\[Language: (\w+)\]")
		for line in codecs.open("indexes/GUTINDEX.ALL", encoding="utf8"):
			line = line.replace(u"\xA0", u" ") # Convert non-breaking spaces to ordinary spaces.

			if inpreamble: # Skip the explanation at the start of the file.
				if "TITLE and AUTHOR" in line and "ETEXT NO." in line:
					inpreamble = False
				else:
					continue

			if not line.strip():
				continue # Ignore empty lines.

			if line.startswith("<==End of GUTINDEX.ALL"):
				break # Done.

			if line.startswith((u" ", u"\t", u"[")):
				# Attribute line; see if it specifies the language.
				m = langre.search(line)
				if m:
					language = m.group(1)
					self.ebookslanguage[ebookno] = language
			else:
				# Possibly title line: "The German Classics     51389"
				parts = line.strip().rsplit(" ", 1)
				if len(parts) < 2:
					continue
				title, ebookno = parts
				title = title.strip()
				try:
					if ebookno.endswith(("B", "C")):
						ebookno = ebookno[:-1]
					ebookno = int(ebookno)
					# It's a genuine title.
					ebooks[ebookno] = title
				except ValueError:
					continue # Missing or invalid ebook number

		# Default language is English; mark every eBook which hasn't a language specified as English.
		for nr, title in ebooks.iteritems():
			if not nr in self.ebookslanguage:
				self.ebookslanguage[nr] = "English"

	'''deprecated'''

	'''
	given a url_reader that yields URLs to Project Gutenberg-format HTML
	pages that contain books, saves each book to a text file.
	'''
	# def html_to_txt(self, url_reader):

	# 	for header_text, url in url_reader.yield_url():
	# 		# get HTML
	# 		html_text = None
	# 		try:
	# 			response = requests.get(url)
	# 			html_text = etree.HTML(response.text)
	# 		except ValueError:
	# 			html_text = html.parse(url)
	# 		except:
	# 			print "unexpected error:", sys.exc_info()[0]

	# 		# process novel body
	# 		string_text = '\n'.join([el.text for el in html_text.findall('.//p')])
	# 		if not isinstance(string_text, basestring):
	# 			string_text = unicodedata.normalize('NFKD', unicode_text).encode('ascii', 'ignore')
	# 		stripped_text = string_text.strip()

	# 		# get title
	# 		novel_name = header_text + " " + html_text.find('.//title').text.strip() + ".txt"

	# 		# write to text file
	# 		text_file = open(novel_name, "w")
	# 		text_file.write(stripped_text)
	# 		text_file.close()


	def book_id_to_txt(self, id_reader):
		"""
		Given a reader that yields (header_text, book_id) tuples,
		downloads book into subdirectories by header_text.

	    Args:
	      id_reader
	    """

		for nr, tup in enumerate(id_reader.yield_id()):
			header_text, ebookno = tup

			# if group directory doesn't exist, create it
			if not os.path.exists("ebooks-zipped/"+header_text):
				os.mkdir("ebooks-zipped/"+header_text)

			if self.ebookslanguage[ebookno] != self.language: # Only fetch books for specified language.
				continue
			filedir = self.mirrordir.get(ebookno)
			filename = self.mirrorname.get(ebookno)
			if not filedir or not filename:
				continue
			url = self.mirror + filedir + "/" + filename

			fn = os.path.join("ebooks-zipped", header_text, filename)

			if os.path.exists(fn):
				print "%d %s exists, download not necessary" % (nr, fn)
			else:
				print "%d downloading %s..." % (nr, fn)
				# Slow with FTP mirrors; prefer a HTTP mirror.
				urllib.urlretrieve(url, fn)


				# Fast, but requires external wget utility.
				# cmd = "wget -O %s %s" % (fn, url)
				# os.system(cmd)
		

		# Unzip them.
		errors = []
		for dir_name, subdir_list, file_list in os.walk("ebooks-zipped/"):
			for fn in file_list:
				zip_file_path = os.path.join(dir_name, fn)
				unzipped_file_dir = os.path.join("ebooks-unzipped", os.path.relpath(dir_name,"ebooks-zipped/"))

				#TODO prevent unzipping if already exists

				print "extracting", fn
				try:
					zipfile.ZipFile(zip_file_path).extractall(unzipped_file_dir)
				except zipfile.BadZipfile:
					errors.append("Error: can't unzip %s" % fn) # Some files in the Gutenberg archive are damaged.


mirror_url = "http://eremita.di.uminho.pt/gutenberg/"
if len(sys.argv) > 1:
	mirror_url = str(sys.argv[1])


#scraper = Scraper("http://www.mirrorservice.org/sites/ftp.ibiblio.org/pub/docs/books/gutenberg/")
# scraper = Scraper("http://eremita.di.uminho.pt/gutenberg/")
scraper = Scraper(mirror_url)
# temporarily read the local html file, since Chris got blocked...
scraper.book_id_to_txt(BookshelfIdReader("bestsellers.html"))
# scraper.book_id_to_txt(BookshelfUrlReader("bestsellers.html"))



