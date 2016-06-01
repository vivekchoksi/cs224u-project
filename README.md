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

Book Metadata

	dict of book_id: metadata, where metadata is a dict of 
		title
		author
		year
		ml_editors_rank
		ml_readers_rank
		pub_prog_rank
		exp_rank
		bestsellers_rank
		gender
		natl
		num_lists

	to their respective values.

	For each of the "list" values, the value is the book's ranking on the list.
	If the value is 0, the book was not on that list. 
