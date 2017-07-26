from gensim.models import KeyedVectors
import numpy as np
import time
import sys, os
import cPickle as pickle
import json
import random

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Stanford-OpenIE-Python'))
from wrapper import stanford_ie, extract_events_filelist

def checkExists(word, model):
  """
  Convenience function to check if a given word is in a word embedding model.
  model = KeyedVectors.load_word2vec_format('../datasets/googlenews-vectors-negative300.bin', binary=True)
  """
  if word in model:
    if len(model[word]) > 0:
      return True
    else:
      return False
  else:
    return False

def getTextFilesInDirectory(dir, recursive = True, ignore = ['.', ',', '..']):
  """
  Recursively walks through a directory, storing all paths to text files.
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

def getEventTriplesFromBatch(batch_file, verbose=False):
  """
  Expects batch_file in format:
  47.['South African Mine', ' Cut', ' Exports']
  Output: [['South African Mine', 'Cut', 'Exports'], ['Elizabeth Amon', 'is in', 'Brooklyn'] ... ]
  """
  triples = []
  unparsable_lines = 0
  with open(batch_file, 'r') as batch:
    for line in batch:
      try:
        words = line[line.find('.')+1:]
        words = words.replace("\n", "")
        words = words[1:len(words)-1]
        words = words.replace("'", "")
        triple = words.split(',')
        for i in range(3):
          while triple[i][0] == " ":
            triple[i] = triple[i][1:]
        triples.append(triple)
      except:
        if verbose:
          print "Line caused exception:", line
        unparsable_lines += 1
  if verbose:
    print "Unparsable lines:", unparsable_lines
  return triples

def loadEventsFromBatchFiles(batch_file_list, return_dicts=True):
  """
  return_dicts: return dictionaries created from the batch files
  Otherwise returns just the events.
  """
  start = time.time()
  subjects = {}
  actions = {}
  predicates = {}
  events = []

  ctr = 0
  sCollisions, aCollisions, pCollisions = 0, 0, 0
  for batch_file in batch_file_list:
    ctr += 1
    print "Loading batch %d/%d into memory." % (ctr, len(batch_file_list))
    batch_file = os.path.abspath(batch_file)
    triples = getEventTriplesFromBatch(batch_file)
    events.extend(triples)

    if (return_dicts):
      for t in triples:
        if t[0] in subjects:
          subjects[t[0]] += 1
          sCollisions += 1
        else:
          subjects[t[0]] = 1

        if t[1] in actions:
          actions[t[1]] += 1
          aCollisions += 1
        else:
          actions[t[1]] = 1

        if t[2] in predicates:
          predicates[t[2]] += 1
          pCollisions += 1
        else:
          predicates[t[2]] = 1

  if (return_dicts):
    print "\n Unique Subjects: %d Unique Actions: %d Unique Predicates: %d" % (len(subjects), len(actions), len(predicates))
    print "\n Subject Collisions: %d Action Collisions: %d Predicate Collisions: %d" % (sCollisions, aCollisions, pCollisions)
    print "Finished in %f secs." % (time.time() - start)
    return (subjects, actions, predicates)
  else:
    return events


def writeRawDictionariesToDisk(subject_dict, action_dict, predicate_dict, dump_dir = './dicts', how='json'):
  """
  Stores the triple dictionaries to pickle files.
  """
  dump_dir = os.path.abspath(dump_dir)
  if not os.path.exists(dump_dir):
    os.makedirs(dump_dir)

  if (how == 'pickle'):
    print "Pickling subject dict..."
    pickle.dump(subject_dict, open(os.path.join(dump_dir, 'subjects.p'), 'wb'))
    print "Pickling action dict..."
    pickle.dump(action_dict, open(os.path.join(dump_dir, 'actions.p'), 'wb'))
    print "Pickling predicate dict..."
    pickle.dump(predicate_dict, open(os.path.join(dump_dir, 'predicates.p'), 'wb'))

  elif (how == 'json'):
    print "JSONing subject dict..."
    with open(os.path.join(dump_dir, 'subjects.json'), 'w') as fp:
      json.dump(subject_dict, fp, indent=0)
    print "JSONing action dict..."
    with open(os.path.join(dump_dir, 'actions.json'), 'w') as fp:
      json.dump(action_dict, fp, indent=0)
    print "JSONing predicate dict..."
    with open(os.path.join(dump_dir, 'predicates.json'), 'w') as fp:
      json.dump(predicate_dict, fp, indent=0)

  else:
    print "Error: serializer type not understood."
  print "Finished storing dictionaries to dump files."

def writeIndexedDictionariesToDisk(subject_dict_raw, action_dict_raw, predicate_dict_raw, dump_dir = './dicts', how='json'):
  """
  Stores subject, action, predicate dictionaries to disk where each item is:
  id (integer): item (string)

  Deletes dictionaries when it's done with them to free up memory.
  """
  dump_dir = os.path.abspath(dump_dir)
  if not os.path.exists(dump_dir):
    os.makedirs(dump_dir)

  idx_subjects_by_id = {}
  idx_actions_by_id = {}
  idx_predicates_by_id = {}

  idx_subjects_by_str = {}
  idx_actions_by_str = {}
  idx_predicates_by_str = {}

  # build subject dictionaries
  s_idx = 0
  for s in subject_dict_raw.keys():
    idx_subjects_by_id[s_idx] = s
    idx_subjects_by_str[s] = s_idx
    s_idx += 1

  print "Dumping indexed subjects dictionaries..."
  if (how == 'pickle'):
    pickle.dump(idx_subjects_by_id, open(os.path.join(dump_dir, 'subjects_by_id.p'), 'wb'))
    pickle.dump(idx_subjects_by_str, open(os.path.join(dump_dir, 'subjects_by_str.p'), 'wb'))
  else:
    with open(os.path.join(dump_dir, 'subjects_by_id.json'), 'w') as fp:
      json.dump(idx_subjects_by_id, fp, indent=0)
    with open(os.path.join(dump_dir, 'subjects_by_str.json'), 'w') as fp:
      json.dump(idx_subjects_by_str, fp, indent=0)
  del idx_subjects_by_id, idx_subjects_by_str, subject_dict_raw

  # build actions dictionaries
  a_idx = 0
  for a in action_dict_raw.keys():
    idx_actions_by_id[a_idx] = a
    idx_actions_by_str[a] = a_idx
    a_idx += 1

  print "Dumping indexed actions dictionaries..."
  if (how == 'pickle'):
    pickle.dump(idx_actions_by_id, open(os.path.join(dump_dir, 'actions_by_id.p'), 'wb'))
    pickle.dump(idx_actions_by_str, open(os.path.join(dump_dir, 'actions_by_str.p'), 'wb'))
  else:
    with open(os.path.join(dump_dir, 'actions_by_id.json'), 'w') as fp:
      json.dump(idx_actions_by_id, fp, indent=0)
    with open(os.path.join(dump_dir, 'actions_by_str.json'), 'w') as fp:
      json.dump(idx_actions_by_str, fp, indent=0)
  del idx_actions_by_id, idx_actions_by_str, action_dict_raw

  # build predicates dictionaries
  p_idx = 0
  for p in predicate_dict_raw.keys():
    idx_predicates_by_id[p_idx] = p
    idx_predicates_by_str[p] = p_idx
    p_idx += 1

  print "Dumping indexed predicates dictionaries..."
  if (how == 'pickle'):
    pickle.dump(idx_predicates_by_id, open(os.path.join(dump_dir, 'predicates_by_id.p'), 'wb'))
    pickle.dump(idx_predicates_by_str, open(os.path.join(dump_dir, 'predicates_by_str.p'), 'wb'))
  else:
    with open(os.path.join(dump_dir, 'predicates_by_id.json'), 'w') as fp:
      json.dump(idx_predicates_by_id, fp, indent=0)
    with open(os.path.join(dump_dir, 'predicates_by_str.json'), 'w') as fp:
      json.dump(idx_predicates_by_str, fp, indent=0)
  del idx_predicates_by_id, idx_predicates_by_str, predicate_dict_raw

  print "Finished storing indexed dictionaries to dump files."


def loadRawDictionaries(dump_dir = './dicts', how='json'):
  """
  Loads dictionaries subjects.json, actions.json, and predicates.json
  """
  dump_dir = os.path.abspath(dump_dir)

  if (how == 'json'):
    with open(os.path.join(dump_dir, 'subjects.json'), 'r') as fp:
      subjects = json.load(fp)
    print "Loaded subjects dictionary."
    with open(os.path.join(dump_dir, 'actions.json'), 'r') as fp:
      actions = json.load(fp)
    print "Loaded actions dictionary."
    with open(os.path.join(dump_dir, 'predicates.json'), 'r') as fp:
      predicates = json.load(fp)
    print "Loaded predicates dictionary."

  elif (how == 'pickle'):
    subjects = pickle.load(os.path.join(dump_dir, 'subjects.p'), "rb")
    print "Loaded subjects dictionary."
    actions = pickle.load(os.path.join(dump_dir, 'actions.p'), "rb")
    print "Loaded actions dictionary."
    predicates = pickle.load(os.path.join(dump_dir, 'predicates.p'), "rb")
    print "Loaded predicates dictionary."

  else:
    print "Error: serializer type not understood."
  print "Finished loading dictionaries from dump."
  return (subjects, actions, predicates)

def loadIndexedDictionaries(dump_dir = './dicts', how='json',
    types=['subjects_by_id', 'subjects_by_str', 'actions_by_id', 'actions_by_str', 'predicates_by_id', 'predicates_by_str']):
  """
  Loads a customizable list of dictionaries.
  By default will load all dictionaries, which can take
  up a large amount of memory (12-16GB)
  """
  dump_dir = os.path.abspath(dump_dir)

  if (how == 'json'):
    result_dicts = {}
    for t in types:
      with open(os.path.join(dump_dir, '%s.json' % t), 'r') as fp:
        result_dicts[t] = json.load(fp)
        print "Loaded %s.json." % t

  elif (how == 'pickle'):
    for t in types:
      result_dicts[t] = pickle.load(os.path.join(dump_dir, '%s.p' % t), 'rb')
      print "Loaded %s.p" % t

  else:
    print "Error: serializer type not understood."

  print "Finished loading dictionaries from dump."
  return result_dicts
  # return (subjects_by_id, actions_by_id, predicates_by_id, subjects_by_str, actions_by_str, predicates_by_str)
 

def extractEvents(corpus_paths, batch_size = 400, filelist_path = '_filelist.txt',
          out_dir = './events', start_batch = 0):
  """
  Extracts events from articles in a directory and writes them to disk.
  corpus_paths - directory to be searched recursively for articles
  batch_size - the number of articles to be processed by StanfordIE at once
  filelist_path - a temporary text file that is used to write down article paths for StanfordIE
  out_dir - the folder where batch files should be stored
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

def getRandom(dict_by_id):
  """
  Given a dictionary that is indexed by id, will select one of the dictionaries
  key-value pairs uniformly at random.

  dict_by_id: a dictionary with uid's as keys and phrases as values
  """
  randId = str(random.randint(0, len(dict_by_id)-1))
  return (randId, dict_by_id[randId])

def writeCorruptEvents(corr_dir='./corrupt', event_prefix ='batch_', event_suffix='.txt',
                       corr_name_format='corrupt_*.txt', real_dir='./real',
                       real_name_format='real_*.txt'):
  """
  For each batch file, writes a corrupted batch file where one of the
  three arguments of the triple is replaced randomly from the corresponding dictionary.
  """
  dicts = loadIndexedDictionaries(types=['subjects_by_id', 'actions_by_id', 'predicates_by_id'])
  subjects_by_id = dicts['subjects_by_id']
  actions_by_id = dicts['actions_by_id']
  predicates_by_id = dicts['predicates_by_id']

  # make corrupt path
  corrPath = os.path.abspath(corr_dir)
  if not os.path.exists(corrPath):
    os.makedirs(corrPath)

  # make real path
  realPath = os.path.abspath(real_dir)
  if not os.path.exists(realPath):
    os.makedirs(realPath)

  batchPaths = getBatchPaths('all')
  for p in batchPaths:
    start = p.find(event_prefix)
    batchNum = p[start+len(event_prefix):]
    end = batchNum.find(event_suffix)
    batchNum = batchNum[:end]
    batchNum = int(batchNum)
    print "Starting batch #%d." % batchNum 

    events = loadEventsFromBatchFiles([p], return_dicts=False)

    corrFile = open(os.path.join(corrPath, corr_name_format.replace('*', str(batchNum))), 'w')
    realFile = open(os.path.join(realPath, real_name_format.replace('*', str(batchNum))), 'w')

    idx = 0
    for e in events:
      if (idx % 10000 == 0):
        print "Finished event %d/%d." % (idx, len(events))

      realFile.write('%d.%s\n' % (idx, e))

      # replace a random arg in the event triple
      randomArg = random.randint(0, 2)
      if (randomArg == 0):
        randId, randPhrase = getRandom(subjects_by_id)
      elif (randomArg == 1):
        randId, randPhrase = getRandom(actions_by_id)
      else:
        randId, randPhrase = getRandom(predicates_by_id)
      e[randomArg] = randPhrase

      corrFile.write('%d.%s\n' % (idx, e))

      idx += 1

    corrFile.close()
    realFile.close()

def getBatchPaths(ids, dir='./events', name_format='batch_*.txt'):
  """
  Returns all of the batch files that match the given name_format and dir.
  ids: A list of batch file numbers, or 'all'
  """
  path = os.path.join(os.path.abspath(dir), name_format)

  if (ids == 'all'):
    paths = getTextFilesInDirectory(dir, recursive=False)
  else:
    paths = []
    for i in ids:
      paths.append(path.replace("*", str(i)))
  return paths

def getEventsFromCorpus():
  """
  Gets all events from articles located in the corpus, and writes these events to disk.
  - Events are written into batch files, i.e events/batch_64.txt
  - Each batch file contains an event on each line in the format:
    47.['South African Mine', ' Cut', ' Exports']
  """
  print ("Starting getEventsFromCorpus...")
  corpus_paths = ['/home/milo/envs/trading/datasets/financial-news-dataset-master/20061020_20131126_bloomberg_news',
                  '/home/milo/envs/trading/datasets/financial-news-dataset-master/ReutersNews106521']
  extractEvents(corpus_paths, start_batch = 0)

def averageEmbedding(string, model, word_embedding_size=300):
  avgEmbed = np.zeros(word_embedding_size)
  words = string.split(' ')
  valid_ctr = 0
  for w in words:
    if (checkExists(w, model) == True):
      valid_ctr += 1
      avgEmbed = np.add(avgEmbed, model[w])
  if valid_ctr > 0:
    return avgEmbed / valid_ctr
  else:
    print "Error: Could not embed any words in phrase:", string
    return False

def getTrainingTensor(num_examples, word_embedding_size=300):
  """
  Constructs an nx6xd tensor where
  n: number of training examples
  6: these six rows are actor, action, object, corrupt_actor, corrupt_action, corrupt_object
  d: the dimension of word embeddings (Google's pretrained model has d=300)

  Will keep getting batch files until num_examples have been found.
  """
  wordEmbeddingModel = KeyedVectors.load_word2vec_format('../datasets/googlenews-vectors-negative300.bin', binary=True)
  print "Loaded word embedding model."

  real_dir = './real'
  corr_dir = './corrupt'
  real_path = os.path.abspath(real_dir)
  corr_path = os.path.abspath(corr_dir)
  real_files = getTextFilesInDirectory(real_path, recursive=False)
  corr_files = getTextFilesInDirectory(corr_path, recursive=False)
  print "Found text files from directories."
  real_files.sort()
  corr_files.sort()
  print "Sorted text files by batch number."

  trainTensor = np.zeros((num_examples, 6, word_embedding_size))

  ctr = 0
  for b in range(len(real_files)):
    realTriples = getEventTriplesFromBatch(real_files[b])
    corrTriples = getEventTriplesFromBatch(corr_files[b])

    for t in range(len(realTriples)):
      row0 = averageEmbedding(realTriples[t][0], wordEmbeddingModel)
      row1 = averageEmbedding(realTriples[t][1], wordEmbeddingModel)
      row2 = averageEmbedding(realTriples[t][2], wordEmbeddingModel)
      row3 = averageEmbedding(corrTriples[t][0], wordEmbeddingModel)
      row4 = averageEmbedding(corrTriples[t][1], wordEmbeddingModel)
      row5 = averageEmbedding(corrTriples[t][2], wordEmbeddingModel)

      for row in [row0, row1, row2, row3, row4, row5]:
        if type(row) is not np.ndarray:
          print "Skipping event due to unknown words."
          continue

      trainTensor[ctr][0] = row0
      trainTensor[ctr][1] = row1
      trainTensor[ctr][2] = row2
      trainTensor[ctr][3] = row3
      trainTensor[ctr][4] = row4
      trainTensor[ctr][5] = row5
      ctr += 1
      if (ctr == num_examples):
        return trainTensor

  return False

def main():
  # rel_dir = './events/batch_1.txt'
  # path = os.path.abspath(rel_dir)
  # triples = getTriplesFromBatch(path)
  # print triples
  #paths = getBatchPaths('all')
  #print "Found paths:", len(paths)
  #s, a, p = loadEventsFromBatchFiles(paths)
  # writeDictionariesToDisk(s, a, p, how='json')
  #subjects_raw, actions_raw, predicates_raw = loadRawDictionaries()
  #writeIndexedDictionariesToDisk(subjects_raw, actions_raw, predicates_raw)
  #encodeEventsFilesRealCorrupted()
  tensor = getTrainingTensor(100)
  print tensor[0]
  print tensor.shape

if __name__ == '__main__':
  main()
