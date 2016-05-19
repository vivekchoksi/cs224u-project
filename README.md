# cs224u-project


Scraper How-To

	$ python data/scraper.py
	list of mirror urls: https://www.gutenberg.org/MIRRORS.ALL
	The scraper will take ~10 minutes to run.

	### IMPORTANT ### Once the scraper has finished running, you must manually
	adjust one file:

	data/ebooks-unzipped/1916/14571.txt
		Change the line "encoding:ASCII" to "encoding:latin-1"

	$ python beautify-books.py

	Now the books are in the ebooks directory, in subdirectories by heading.