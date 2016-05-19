'''
scrapes Project Gutenberg for its American bestsellers 1895-1923
list of books
'''

from lxml import html, etree
from bs4 import BeautifulSoup
import cPickle as pickle
import unicodedata
import requests


class UrlReader():
	def yield_url(self):
		# code to yield url of next (Gutenberg) HTML file
		yield "https://www.gutenberg.org/files/327/327-h/327-h.htm"


class Scraper():

	'''
	given a url_reader that yields URLs to Project Gutenberg-format HTML
	pages that contain books, saves each book to a text file.
	'''
	def html_to_txt(self, url_reader):

		for url in url_reader.yield_url():
			# get HTML
			html_text = None
			try:
				response = requests.get(url)
				html_text = etree.HTML(response.text)
			except ValueError:
				html_text = html.parse(url)

			# process novel body
			unicode_text = '\n'.join([el.text for el in html_text.findall('.//p')])
			string_text = unicodedata.normalize('NFKD', unicode_text).encode('ascii', 'ignore')
			stripped_text = string_text.strip()

			# get title
			novel_name = html_text.find('.//title').text.strip() + ".txt"

			# write to text file
			text_file = open(novel_name, "w")
			text_file.write(stripped_text)
			text_file.close()
			

scraper = Scraper()
scraper.html_to_txt(UrlReader())


