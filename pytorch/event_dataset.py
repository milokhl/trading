from __future__ import print_function, division
import os
import torch
from torch import FloatTensor
from torch.utils.data import Dataset, DataLoader
import numpy as np

class EventDataset(Dataset):
    def __init__(self, tensor_dir='input_tensors/', memory_limit=1e9):
        """
        A dataset loader with caching to stay within memory limits.
        Whenever an index outside of the cache is requested, the cache
        is refilled.

        tensor_dir: directory containing .npy files
        memory_limit: how much memory in bytes can this dataset occupy?

        Tips:
        - with a pytorch dataset loader, only use 1 worker
        - make sure that all files in the directory have same number of examples in them
        """
        self.tensor_dir = tensor_dir
        self.tensor_files = getTextFilesInDirectory(self.tensor_dir)

        assert len(self.tensor_files) > 0, 'Error: no suitable files found in directory!'
        test_tensor = np.load(self.tensor_files[0])

        shape = np.shape(test_tensor)
        assert len(shape) == 3, 'Error: input tensor was not 3 dimensional!'

        self.file_batch_size = shape[0]
        self.bytes_per_file = test_tensor.nbytes

        self.files_in_cache = int(memory_limit / self.bytes_per_file)
        self.items_in_cache = self.files_in_cache * self.file_batch_size
        self.cache = np.zeros((self.items_in_cache, shape[1], shape[2]))

        self.cache_low_idx = 0 # The index of the first example in cache.
        self.cache_high_idx = self.files_in_cache * self.file_batch_size - 1 # Index of the last example in the cache.

        self.fill_cache(0)
        print('------- Initialized event dataset! ------- ')
        print('# Files:', len(self.tensor_files))
        print('# per file:', self.file_batch_size)
        print('# Files resident:', self.files_in_cache)
        print('# Items resident:', self.items_in_cache)

    def fill_cache(self, idx):
        """ Fills the cache starting with the lowest file that contains example idx. """
        print('Refilling cache from:', idx)
        file_idx = idx // self.file_batch_size

        # Fully fill the cache, or stop once all files have been used up.
        for ii in range(min(self.files_in_cache, len(self.tensor_files)-file_idx)):
            low_idx = ii*self.file_batch_size
            high_idx = (ii+1)*self.file_batch_size
            self.cache[low_idx:high_idx,:,:] = np.load(self.tensor_files[file_idx + ii])

        self.cache_low_idx = file_idx * self.file_batch_size
        self.cache_high_idx = self.cache_low_idx + self.items_in_cache

    def __len__(self):
        """ Assumes every .npy tensor file has the same batch size in it. """
        return self.file_batch_size * len(self.tensor_files)

    def __getitem__(self, idx):

        # If the index is not in the cache, refill cache.
        if idx > self.cache_high_idx or idx < self.cache_low_idx:
            self.fill_cache(idx)
        
        return FloatTensor(self.cache[idx % self.items_in_cache].astype(np.float32))


def getTextFilesInDirectory(dir, recursive=False, ignore = ['.', ',', '..', 'README.md']):
  """ Recursively walks through a directory, storing all absolute paths to text files. """
  res = []
  dir = os.path.abspath(dir)
  for file in os.listdir(dir):
    if not file in ignore:
      new = os.path.join(dir, file)
      if os.path.isdir(new) and recursive:
        subdir_res = getTextFilesInDirectory(new, recursive=recursive, ignore=ignore)
        res.extend(subdir_res)
      else:
        res.append(new)
  return res