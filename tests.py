import unittest
from dataset import getTextFilesInDirectory, getEventTriplesFromBatch
import os
import random
import numpy as np
import time
import keras.backend as K
from model.bilinear_tensor_product import BilinearTensorProduct

class TestDatasetCorrespondence(unittest.TestCase):

  def setUp(self):
    real_dir = './real'
    corr_dir = './corrupt'
    real_path = os.path.abspath(real_dir)
    corr_path = os.path.abspath(corr_dir)
    self.real_files = getTextFilesInDirectory(real_path)
    self.corr_files = getTextFilesInDirectory(corr_path)

  def test_num_batches(self):
    self.assertEqual(len(self.real_files), len(self.corr_files))

  def test_num_events(self):
    self.real_files.sort()
    self.corr_files.sort()
    for i in range(len(self.real_files)):
      rf = open(self.real_files[i])
      cf = open(self.corr_files[i])
      rlen = sum(1 for line in rf)
      clen = sum(1 for line in cf)
      rf.close()
      cf.close()
      self.assertEqual(rlen, clen)

  def test_get_event_triples_from_batch(self):
    self.real_files.sort()
    self.corr_files.sort()
    for trial in range(5):
      i = random.randint(0, len(self.real_files))
      rf = self.real_files[i]
      cf = self.corr_files[i]
      realTriples = getEventTriplesFromBatch(rf)
      corrTriples = getEventTriplesFromBatch(cf)
      self.assertEqual(len(realTriples), len(corrTriples))

  def test_random_similarity(self):
    self.real_files.sort()
    self.corr_files.sort()
    for trial in range(10):
      i = random.randint(0, len(self.real_files))
      rf = self.real_files[i]
      cf = self.corr_files[i]
      realTriples = getEventTriplesFromBatch(rf)
      corrTriples = getEventTriplesFromBatch(cf)
      rand = random.randint(0, len(realTriples))
      real = realTriples[rand]
      corr = corrTriples[rand]
      self.assertTrue(real[0]==corr[0] or real[1]==corr[1] or real[2]==corr[2])

if __name__ == '__main__':
    unittest.main()