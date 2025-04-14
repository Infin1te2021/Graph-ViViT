import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

class BaseDataLoader(DataLoader):
  """
  Base class for all data loaders
  """
  def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate, persistent_workers=False):
    self.validation_split = validation_split
    self.shuffle = shuffle

    self.batch_idx = 0
    self.n_samples = len(dataset)

    self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

    self.init_kwargs = {
      'dataset': dataset,
      'batch_size': batch_size,
      'shuffle': self.shuffle,
      'collate_fn': collate_fn,
      'num_workers': num_workers,
      'persistent_workers': persistent_workers
    }
    super().__init__(sampler=self.sampler, **self.init_kwargs)

  def _split_sampler(self, split):
    if split == 0.0:
      return None, None

    idx_full = np.arange(self.n_samples)

    np.random.seed(0)
    np.random.shuffle(idx_full)

    if isinstance(split, int):
      assert split > 0
      assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
      len_valid = split
    else:
      len_valid = int(self.n_samples * split)

    valid_idx = idx_full[0:len_valid]
    train_idx = np.delete(idx_full, np.arange(0, len_valid))

    # valid_idx = idx_full[len_valid * fold_idx : len_valid * (fold_idx + 1)]
    # train_idx = np.delete(idx_full, np.arange(len_valid * fold_idx, len_valid * (fold_idx + 1)))

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # turn off shuffle option which is mutually exclusive with sampler
    self.shuffle = False
    self.n_samples = len(train_idx)

    return train_sampler, valid_sampler

  def split_validation(self):
    if self.valid_sampler is None:
      return None
    else:
      return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)

# class NTURGBDDataLoader(DataLoader):
#   """
#   NTU-RGB+D and NTU-RGB+D 120 data loader
#   """
#   def __init__(self, dataset, benchmark_evaluation, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
#     self.benchmark_evaluation = benchmark_evaluation
#     self.validation_split = validation_split
#     self.shuffle = shuffle

#     self.batch_idx = 0
#     self.n_samples = len(dataset)

#     self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

#     self.init_kwargs = {
#       'dataset': dataset,
#       'batch_size': batch_size,
#       'shuffle': self.shuffle,
#       'collate_fn': collate_fn,
#       'num_workers': num_workers
#     }
#     super().__init__(sampler=self.sampler, **self.init_kwargs)

#   def _split_sampler(self, validation_split):
#     # Split the dataset into training and test according to benchmark_evaluation
#     training_files, testing_files = train_test_split(self.benchmark_evaluation)
#     test_idx = np.array([i for i, file in enumerate(self.dataset) if file in testing_files])
#     train_idx = np.array([i for i, file in enumerate(self.dataset) if file in training_files])
    
#     test_sampler = SubsetRandomSampler(test_idx)
#     self.n_samples = len(train_idx)

#     # Split the training set into training and validation sets
#     if validation_split == 0.0:
#       return SubsetRandomSampler(train_idx), None, test_sampler
    
#     idx_full = np.arange(self.n_samples)
#     np.random.seed(0)
#     np.random.shuffle(idx_full)
    
#     if isinstance(validation_split, int):
#       assert validation_split > 0
#       assert validation_split < self.n_samples, "validation set size is configured to be larger than training dataset."
#       len_valid = validation_split
#     else:
#       len_valid = int(self.n_samples * validation_split)

#     valid_idx = idx_full[0:len_valid]
#     train_idx = np.delete(idx_full, np.arange(0, len_valid))

#     train_sampler = SubsetRandomSampler(train_idx)
#     valid_sampler = SubsetRandomSampler(valid_idx)

#     self.shuffle = False
#     self.n_samples = len(train_idx)

#     return train_sampler, valid_sampler, test_sampler
  
#   def split_validation(self):
#     if self.valid_sampler is None:
#       return None
#     else:
#       return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
    
#   def split_test(self):
#     if self.test_sampler is None:
#       return None
#     else:
#       return DataLoader(sampler=self.test_sampler, **self.init_kwargs)
