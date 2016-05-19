# cs224u-project


Scraper How-To

	$ python data/scraper.py

	### IMPORTANT ### Once the scraper has finished running, you must manually
	adjust one file:

	data/ebooks-unzipped/1916/14571.txt

	Change the line "encoding:ASCII" to "encoding:latin-1"

	$ python beautify-books.py

	Now the books are in the ebooks directory, in subdirectories by heading.