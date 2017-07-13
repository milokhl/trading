from gensim.models import KeyedVectors
import numpy as np
import params
import time

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Stanford-OpenIE-Python'))
from wrapper import stanford_ie, extract_events_filelist

# model = KeyedVectors.load_word2vec_format('../datasets/googlenews-vectors-negative300.bin', binary=True)

def checkExists(word, model):
	if len(model[word]) > 0:
		return True
	else:
		return False

# def extractArticleTriples(text_file):
# 	"""
# 	Removes extraneous info from article and then parses for event triples.
# 	"""
# 	triples = stanford_ie(text_file)
# 	return triples

# def extractRawEvents(article_paths, out_path, chunk_size=50000, min_occurrences=1):
# 	"""
# 	For 
# 	Extracts raw event triples from articles.
# 	Writes to file in format:
# 	id#.(subject, action, predicate)
# 	"""
# 	results = {}

# 	tmp_path = 'concat.txt'
# 	tmp_path = os.path.abspath(tmp_path)

# 	num_articles = len(article_paths)
# 	print "Concatenating %d articles..." % num_articles

# 	ctr = 0
# 	for in_path in article_paths:
# 		ctr += 1

# 		mode = 'a' # 'w' to overwrite
# 		with open(in_path, 'r') as in_file:

# 			with open(tmp_path, mode) as tmp_file:

# 				# if time to process a chunk
# 				if ctr % chunk_size == 0 and ctr > 0:
# 						tmp_file.write(in_file.read())
# 						tmp_file.write('\n')
# 						triples = stanford_ie(tmp_path)
# 						print "Found %d triples from %d articles." % (len(triples), chunk_size)
# 						for t in triples:
# 							t_str = str(t)
# 							if t_str in results:
# 								results[t_str] += 1
# 							else:
# 								results[t_str] = 1
# 						mode = 'w'

# 				# just append to file
# 				else:
# 					mode = 'a'

# 	print ("Writing the event dictionary to disk...")
# 	with open(out_path, 'w') as out_file:
# 		idx = 0
# 		for key, count in results.iteritems():
# 			if count > min_occurrences:
# 				out_file.write("%d.%s\n" % (idx, key))
# 				idx += 1

# 	return results

def getTextFilesInDirectory(dir, recursive = True, ignore = ['.', ',', '..']):
	"""
	Recursively walks in a top-down manner, storing all paths to text files.
	"""
	res = []
	dir = os.path.abspath(dir)
	for file in os.listdir(dir):
		if not file in ignore:
			new = os.path.join(dir, file)
			if os.path.isdir(new) and recursive:
				subdir_res = getTextFilesInDirectory(new, recursive, ignore)
				res.extend(subdir_res)
			else:
				res.append(new)
	return res

def extractEvents(corpus_paths, batch_size = 400, filelist_path = 'filelist.txt',
									out_dir = './events', start_batch = 0):
	"""
	Extracts events from articles in a directory and writes them to disk.
	:corpus_paths - directory to be searched recursively for articles
	:batch_size - the number of articles to be processed by StanfordIE at once
	:filelist_path - a temporary text file that is used to write down article paths for StanfordIE
	:out_dir - the folder where batch files should be stored
	"""
	article_paths = []
	for cpath in corpus_paths:
		print "[INFO] Extracting articles recursively from path:", cpath
		article_paths.extend(getTextFilesInDirectory(cpath))

	print "[INFO] Total articles found:", len(article_paths)
 
	filelist_path = os.path.abspath(filelist_path) # where article paths are stored (overwritten each batch)
	out_dir = os.path.abspath(out_dir) # a folder where batch files are written to

	ctr = 0
	batch_num = 0
	mode = 'a'
	batch_start = time.time()
	for p in article_paths:

		# if batch complete, write to output file
		if (ctr % batch_size == 0 and ctr > 0):
			batch_start = time.time()
			batch_num += 1
			if (batch_num < start_batch):
				continue

			print "[INFO] Writing event batch #%d. %d/%d articles done." % (batch_num, ctr, len(article_paths))

			# get event tuples from the files listed currently and write them to a batch file
			print "[INFO] Stanford IE is extracting event triples..."
			events = extract_events_filelist(filelist_path, verbose = True, max_entailments_per_clause = 100)
			out_path = os.path.join(out_dir, 'batch_%d.txt' % batch_num)
			e_ctr = 0
			with open(out_path, 'w') as out_file:
				print "[INFO] Writing event triples to batch file."
				for e in events:
					out_file.write("%d.%s\n" % (e_ctr, str(e)))
					e_ctr += 1
			print "[INFO] Finished batch in %f sec.\n -----" % (time.time() - batch_start)

			# now write latest article to the filelist, overwriting
			mode = 'w'
			with open(filelist_path, mode) as filelist:
				filelist.write(p)
				filelist.write('\n')
			mode = 'a'

		# append to end of filelist
		else:
			with open(filelist_path, mode) as filelist:
				filelist.write(p)
				filelist.write('\n')

		ctr += 1

	print "[INFO] Finished writing events to disk!"
	return True

def main():
	print ("Starting...")

	# paths to news source folders
	corpus_paths = ['/home/milo/envs/trading/datasets/financial-news-dataset-master/20061020_20131126_bloomberg_news',
									'/home/milo/envs/trading/datasets/financial-news-dataset-master/ReutersNews106521']
	
	extractEvents(corpus_paths, start_batch = 44)

if __name__ == '__main__':
	main()
