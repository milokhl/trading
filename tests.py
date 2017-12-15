import os, random, time, unittest
import numpy as np

from dataset import getTextFilesInDirectory, getEventTriplesFromBatch

from torch.autograd import gradcheck, Variable
import torch.nn as nn
import torch.optim as optim
from pytorch.bilinear_tensor_layer import *

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

class TestBilinearTensorLayer(unittest.TestCase):
  def test_01(self):
    n = 10
    k = 8
    module = BilinearTensorLayer(n, k)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(module.parameters(), lr=0.001, momentum=0.9, nesterov=True)

    vec1 = Variable(torch.FloatTensor(np.arange(1,n+1).reshape(1,n).astype(np.float32)))
    vec2 = Variable(torch.FloatTensor(-1 * np.arange(1,n+1).reshape(1,n).astype(np.float32)))
    target = Variable(torch.FloatTensor(np.ones(k).reshape(1,k).astype(np.float32)))

    out = module(vec1, vec2)
    loss = criterion(out, target)
    loss.backward()
    optimizer.step()

if __name__ == '__main__':
    unittest.main()