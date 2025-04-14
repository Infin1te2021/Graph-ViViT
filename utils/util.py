import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt

def ensure_dir(dirname):
  dirname = Path(dirname)
  if not dirname.is_dir():
    dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
  fname = Path(fname)
  with fname.open('rt') as handle:
    return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
  fname = Path(fname)
  with fname.open('wt') as handle:
    json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
  ''' wrapper function for endless data loader. '''
  for loader in repeat(data_loader):
    yield from loader

def prepare_device(n_gpu_use):
  """
  setup GPU device if available. get gpu device indices which are used for DataParallel
  """
  n_gpu = torch.cuda.device_count()
  if n_gpu_use > 0 and n_gpu == 0:
    print("Warning: There\'s no GPU available on this machine,"
          "training will be performed on CPU.")
    n_gpu_use = 0
  if n_gpu_use > n_gpu:
    print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
          "available on this machine.")
    n_gpu_use = n_gpu
  device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
  list_ids = list(range(n_gpu_use))
  return device, list_ids


class MetricTracker:
  def __init__(self, *keys, writer=None):
    self.writer = writer
    self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
    self.reset()

  def reset(self):
    for col in self._data.columns:
      self._data.loc[:, col] = 0

  def update(self, key, value, n=1):
    if self.writer is not None:
      self.writer.add_scalar(key, value)
    self._data.loc[key, 'total'] += value * n
    self._data.loc[key, 'counts'] += n
    self._data.loc[key, 'average'] = self._data.loc[key, 'total'] / self._data.loc[key, 'counts']

  def avg(self, key):
    return self._data.loc[key, 'average']

  def result(self):
    return dict(self._data['average'])

class CSVImageProcessor:
  def __init__(self, csv_file):
    self.csv_file = csv_file
    self.data = None
    self.normalized_data = None

  def load_and_normalize_csv_to_grayscale(self):
    data = pd.read_csv(self.csv_file, header=None)
    data = data.to_numpy()

    # Exclude -inf and inf by creating a mask
    valid_mask = np.isfinite(data)  # Mask to keep only finite values
    valid_data = data[valid_mask]  # Extract only valid (finite) data

    # Check if valid data exists
    if valid_data.size == 0:
      raise ValueError("The dataset contains no finite values.")

    # Normalize the valid data to the range [0, 255]
    data_min = np.min(valid_data)
    data_max = np.max(valid_data)
    self.normalized_data = ((data - data_min) / (data_max - data_min)) * 255

    # Set invalid values (inf, -inf) in the original array to 0 (black in grayscale)
    self.normalized_data[~valid_mask] = 0

  def display_image(self, colormap='viridis'):
    if self.normalized_data is None:
      raise ValueError("Data not loaded and normalized. Call load_and_normalize_csv_to_grayscale first.")
    # Determine the size of the data (max x and max y in terms of range)
    max_y, max_x = self.normalized_data.shape  # Rows are Y-axis, Columns are X-axis

    # Display the image 
    plt.imshow(self.normalized_data, cmap=colormap, aspect='auto', extent=[0, max_x, max_y, 0])
    # plt.axis('off')
    plt.colorbar(label='Normalized Value')

    # Add labels and title
    plt.title(f"Rearranged Data Visualization")

    plt.show()

from torch.optim.lr_scheduler import _LRScheduler
class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, total_epoch, after_scheduler=None):
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = -1
        super().__init__(optimizer)

    def get_lr(self):
        return [base_lr * (self.last_epoch + 1) / self.total_epoch for base_lr in self.base_lrs]

    def step(self, epoch=None, metric=None):
        if self.last_epoch >= self.total_epoch - 1:
            if metric is None:
                return self.after_scheduler.step(epoch)
            else:
                return self.after_scheduler.step(metric, epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)

class EdgeIndexGenerator:
  def __init__(self, num_nodes=25):
    self.num_nodes = num_nodes
    self.connections = {
      "left_arm": [[4,5],[5,6],[6,7],[7,21],[7,22]],
      "right_arm": [[8,9],[9,10],[10,11],[11,23],[11,24]],
      "left_leg": [[12,13],[13,14],[14,15]],
      "right_leg": [[16,17],[17,18],[18,19]],
      "torso": [[0,1],[1,20],[2,20],[2,3],[0,12],[0,16],[20,4],[20,8]]
    }

  def generate_edge_index(self):
    adj = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
    for group in self.connections.values():
      for i, j in group:
        adj[i, j] = 1
        adj[j, i] = 1
    
    # Convert the adjacency matrix to edge index
    edge_index = torch.nonzero(torch.tensor(adj), as_tuple=False).T
    return edge_index
  