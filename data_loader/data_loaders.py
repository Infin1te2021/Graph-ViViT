import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from base import BaseDataLoader
import glob
import os
import gzip
import numpy as np
import math
import torch.nn.functional as F

class ProcessNegativeInf:
  """
  Replace -inf values with specific value and pad the sequence to a fixed length.
  """
  def __init__(self, target_length=300, replacement_value=0):
    self.target_length = target_length
    self.replacement_value = replacement_value
  
  def __call__(self, sample):
    current_frame = sample.size(1)
    pad_size = self.target_length - current_frame

    if pad_size > 0:
      pad = (0, 0, 0, 0, 0, pad_size, 0, 0)
      sample = F.pad(sample, pad, value=self.replacement_value)
    elif pad_size < 0:
      sample = sample

    mask = torch.isneginf(sample) | (sample == torch.tensor(-1e9, dtype=sample.dtype, device=sample.device))
    replacement = torch.tensor(self.replacement_value, dtype=sample.dtype, device=sample.device)
    return torch.where(mask, replacement, sample)


class trimmer:
  def __init__(self, designed_len=64):
    self.designed_len = designed_len

  def __call__(self, sample):
    sample = sample[:, :self.designed_len, :, :]
    return sample
    
class uniform_sampler:
  def __init__ (self, target_length=100):
    self.target_length = target_length

  def __call__(self, x):
    C, T, H, W = x.shape
    assert T >= self.target_length, f"Input frames ({T}) must be >= target length ({self.target_length})"

    indices = self._uniform_sample(T, self.target_length, x.device)
    x = x[:, indices, :, :]
    return x
  
  @staticmethod
  def _uniform_sample(original_length, target_length, device='cpu'):
    """
    生成均匀采样的索引
    Args:
        original_length (int): 原始帧数 T
        target_length (int): 目标帧数 M
        device: 生成的索引所在设备
    """
    split_sizes = torch.full(
        (target_length,), 
        original_length // target_length, 
        dtype=torch.long, 
        device=device
    )
    remainder = original_length % target_length
    if remainder > 0:
        split_sizes[:remainder] += 1  # 将余数分配到前几个区间
    
    indices = []
    current = 0
    for size in split_sizes:
        end = current + size
        # 在区间 [current, end) 内随机采样一帧
        idx = torch.randint(low=current, high=end, size=(1,), device=device)
        indices.append(idx)
        current = end
    
    return torch.cat(indices)

class KeepLeftThree:
  def __call__(self, sample):
    return sample[:, :, :, :3]

class KeepRightThree:
  def __call__(self, sample):
    return sample[:, :, :, 3:]

class ReshapeChannel3With2BodyParts:
  def __call__(self, sample):
    sampleL = sample[:, :, :, :3]
    sampleR = sample[:, :, :, 3:]

    # if torch.all(sampleR == sampleL):
    #   sampleR = torch.zeros_like(sampleL)
      # sampleR = torch.full_like(sampleL, -1e9)

    sampleL_processed = sampleL.squeeze(0).permute(2, 0, 1).unsqueeze(-1)
    sampleR_processed = sampleR.squeeze(0).permute(2, 0, 1).unsqueeze(-1)
    sample = torch.cat([sampleL_processed, sampleR_processed], dim=-1)
    ## [C=3, T, V, M=2]
    return sample
  
class MinMaxNormalization:
  def __call__(self, sample):
    # min_val = sample.view(sample.size(0), -1).min(dim=1).values.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    min_val = sample.view(sample.size(0), -1).min(dim=1, keepdim=True)[0]
    max_val = sample.view(sample.size(0), -1).max(dim=1, keepdim=True)[0]

    min_val = min_val.view(-1, 1, 1, 1)
    max_val = max_val.view(-1, 1, 1, 1)

    ranges = max_val - min_val
    ranges[ranges == 0] = 1

    sample = (sample - min_val) / ranges
    sample = sample * 2 - 1

    return sample

class MotionDifference:
  def __call__(self, sample):
    motion = sample[:, 1:, :, :]
    motion = motion - sample[:, :-1, :, :]
    last_row = torch.zeros_like(sample[:, -1:, :, :])
    motion = torch.cat((motion, last_row), dim=1)
    sample = motion
    # sample = torch.cat((sample, motion), dim=3)
    return sample

class FramePyramidPool:
  def __init__(self, levels=[64,16,4,1], mode='max'):
    self.levels = levels
    self.mode = mode
  
  def __call__(self, sample):
    np.random.seed(412)
    tpp = []
    original_sample = sample.clone()

    for level in self.levels:
      sample = original_sample.clone()
      frame_num = sample.size(dim=1)
      pad = level * (math.ceil(frame_num / level)) - frame_num

      while pad > (frame_num-1):
        interpolation = (sample[:, 1:, :, :] + sample[:, :-1, :, :]) / 2
        expanded_sample = torch.zeros((sample.size(0), (2*frame_num)-1, sample.size(2), sample.size(3)))
        expanded_sample[:, 0::2, :, :] = sample
        expanded_sample[:, 1::2, :, :] = interpolation

        sample = expanded_sample
        pad = pad - frame_num + 1

        frame_num = sample.size(dim=1)

      if pad <= (frame_num-1):
        indices = np.random.choice(frame_num - 1, size=pad, replace=False)
        for idx in sorted(indices, reverse=True):
          interpolation = (sample[:, idx + 1, :, :] + sample[:, idx, :, :]) / 2
          sample = torch.cat((sample[:, :idx + 1, :, :], interpolation.unsqueeze(1), sample[:, idx + 1:, :, :]), dim=1)

        frame_num = sample.size(dim=1)

      partition_size = frame_num / level
      parts = torch.chunk(sample, level, dim=1)
      pooled_results = []

      for part in parts:
        if self.mode == "avg":
          pooled_results.append(part.mean(dim=1, keepdim=True).float())
        elif self.mode == "max":
          pooled_results.append(part.max(dim=1, keepdim=True).values.float())
        else:
          raise ValueError(f"Unsupported pooling mode: {self.mode}")
        
      tpp.append(torch.cat(pooled_results, dim=1))

    for pooled in tpp:
      tpp_concatenated = torch.cat(tpp, dim=1)
    
    sample = tpp_concatenated

    return sample

class HybridNetPreNormalizationTransform:
  def __init__ (self, zaxis=[0,1], xaxis=[8,4]):
    self.zaxis = zaxis
    self.xaxis = xaxis
  
  def __call__(self, sample):
    np_data = sample.numpy()[np.newaxis, ...]
    processed_data = self.pre_normalization(np_data, self.zaxis, self.xaxis)
    return torch.from_numpy(processed_data[0])
  
  def pre_normalization(self, data, zaxis, xaxis):
      N, C, T, V, M = data.shape
      s = np.transpose(data, [0, 4, 2, 3, 1])  # [N, M, T, V, C]

      # Step 0: Fill in empty frames (Mute if not necessary
      for i_s in range(N):
          skeleton = s[i_s]  # [M, T, V, C]
          if skeleton.sum() == 0:
              continue
              
          for i_p in range(M):
              person = skeleton[i_p]  # [T, V, C]
              if person.sum() == 0:
                  continue
              
              # 处理首帧为空的特殊情况
              if person[0].sum() == 0:
                  valid_frames = person.sum(axis=(1,2)) != 0
                  if valid_frames.any():
                      first_valid = np.where(valid_frames)[0][0]
                      person[:first_valid+1] = person[first_valid]
              
              # 前向填充空帧
              last_non_zero = person[0]
              for t in range(T):
                  if person[t].sum() == 0:
                      s[i_s, i_p, t] = last_non_zero
                  else:
                      last_non_zero = person[t]

      # Step 1: Subtract center joint
      for i_s in range(N):
          skeleton = s[i_s]
          if skeleton.sum() == 0:
              continue
          main_body_center = skeleton[0][:, 1:2, :].copy()  # [T, 1, C]
          for i_p in range(M):
              person = skeleton[i_p]
              if person.sum() == 0:
                  continue
              mask = (person.sum(axis=-1) != 0).reshape(T, V, 1)
              s[i_s, i_p] = (person - main_body_center) * mask

      # Step 2: Align z-axis
      for i_s in range(N):
          skeleton = s[i_s]
          if skeleton.sum() == 0:
              continue
          joint_bottom = skeleton[0, 0, zaxis[0]]
          joint_top = skeleton[0, 0, zaxis[1]]
          vec = joint_top - joint_bottom
          if np.linalg.norm(vec) < 1e-6:
              continue  # 避免零向量
          axis = np.cross(vec, [0, 0, 1])
          angle = self.angle_between(vec, [0, 0, 1])
          matrix_z = self.rotation_matrix(axis, angle)
          for i_p in range(M):
              person = skeleton[i_p]
              if person.sum() == 0:
                  continue
              for t in range(T):
                  for v in range(V):
                      s[i_s, i_p, t, v] = np.dot(matrix_z, person[t, v])

      # Step 3: Align x-axis
      for i_s in range(N):
          skeleton = s[i_s]
          if skeleton.sum() == 0:
              continue
          joint_rshoulder = skeleton[0, 0, xaxis[0]]
          joint_lshoulder = skeleton[0, 0, xaxis[1]]
          vec = joint_rshoulder - joint_lshoulder
          if np.linalg.norm(vec) < 1e-6:
              continue  # 避免零向量
          axis = np.cross(vec, [1, 0, 0])
          angle = self.angle_between(vec, [1, 0, 0])
          matrix_x = self.rotation_matrix(axis, angle)
          for i_p in range(M):
              person = skeleton[i_p]
              if person.sum() == 0:
                  continue
              for t in range(T):
                  for v in range(V):
                      s[i_s, i_p, t, v] = np.dot(matrix_x, person[t, v])

      sample = np.transpose(s, [0, 4, 2, 3, 1])  # [N, C, T, V, M]
      return sample
  
  @staticmethod
  def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    if np.abs(axis).sum() < 1e-6 or np.abs(theta) < 1e-6:
        return np.eye(3)
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
  
  @staticmethod
  def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    if np.abs(v1).sum() < 1e-6 or np.abs(v2).sum() < 1e-6:
        return 0
    v1_u = v1 / v1_norm
    v2_u = v2 / v2_norm
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
  

#class NTURGBDDataset(Dataset):
#  """
#  NTU RGB+D Dataset
#  """
#  def __init__(self, pt_dir, train=True, transforms=None):
#    """
#    Args:
#      data_dir (str): Directory where the .pt files are saved
#      task (str): 'cross-subject' or 'cross-view' for the specific task
#      train (bool): If True, load training data; otherwise, load test data
#      transform (callable, optional): Transformations to apply to data
#    """
#    self.data_dir = pt_dir
#    self.transform = transforms
#    self.train = train
#    
#    self.task = self.data_dir.split("/")[-1]
#    # Determine file path based on the split
#    split = "train" if self.train else "test"
#    self.data = {}
#    checkpoint_files = glob.glob(os.path.join(self.data_dir, f"{self.task}_{split}_*.pt.gz"))
#    print(f"Loading data from files: {checkpoint_files}")
#
#    # Load and decompress data
#    for file in checkpoint_files:
#      with gzip.open(file, 'rb') as f:
#        checkpoint_data = torch.load(f, weights_only=False)
#      self.data.update(checkpoint_data)  # Merge all checkpoint data into a single dictionary
#
#    # List all files (keys) in the combined dataset
#    self.files = list(self.data.keys())
#
#  def __len__(self):
#    return len(self.files)
#
#  def __getitem__(self, idx):
#    file = self.files[idx]
#    sample = self.data[file]
#    data, label = sample['data'], sample['label']
#    label = int(label) - 1
#
#
#    if self.transform:
#      data = self.transform(data)
#  
#    return data, label
#
#class NTUDataLoader(BaseDataLoader):
#  """
#  NTU data loading
#  """
#  def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=10, training=True, persistent_workers=False):
#    # pyramid_levels = [128, 64, 32, 16]
#    # pyramid_pool_mode = 'avg'
#
#    trsfm = transforms.Compose([
#
#      ProcessNegativeInf(target_length=64, replacement_value=0),
#
#      ReshapeChannel3With2BodyParts(),
#
#      uniform_sampler(target_length=64),
#
#      ## For ST-GCN and HybridNet
#      HybridNetPreNormalizationTransform(),
#
#      # MotionDifference(),
#      # FramePyramidPool(
#      #   levels=pyramid_levels,
#      #   mode=pyramid_pool_mode
#      # ),
#    ])
#
#    self.data_dir = data_dir
#    self.training = training  # Whether to load training data or not
#
#    self.batch_size = batch_size
#    self.shuffle = shuffle
#    self.validation_split = validation_split
#    self.num_workers = num_workers
#    self.persistent_workers = persistent_workers
#
#    self.dataset = NTURGBDDataset(pt_dir=self.data_dir, train=self.training, transforms=trsfm)
#    super().__init__(
#      self.dataset,
#      batch_size=self.batch_size,
#      shuffle=self.shuffle,
#      validation_split=self.validation_split,
#      num_workers=self.num_workers,
#      persistent_workers=self.persistent_workers
#    )


import data_loader.tools as tools
import random 
class NPZNTURGBDDataset(Dataset):
    def __init__(self, data_pth, train=True, transforms=None, p_interval=[0.5, 1], aug_method='a123489', intra_p=0.5, inter_p=0.2, max_frame=64, num_workers=56):
      
      self.data_pth = data_pth
      self.train = train
      self.transforms = transforms

      self.aug_method = aug_method
      self.intra_p = intra_p
      self.inter_p = inter_p
      self.max_frame = max_frame
      self.p_interval = p_interval
      self.num_workers = num_workers

      npz_data = np.load(self.data_pth)
      if self.train:
          self.data = npz_data['x_train']
          self.label = np.where(npz_data['y_train'] > 0)[1]
          self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
      else:
          self.data = npz_data['x_test']
          self.label = np.where(npz_data['y_test'] > 0)[1]
          self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
      
      N, T, _ = self.data.shape
      self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)

      self.valid_frame_nums = []
      self.num_peoples = []
      for i in range(len(self.data)):
        data_numpy = self.data[i]
        self.valid_frame_nums.append(np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0))
        self.num_peoples.append(np.sum(data_numpy.sum(0).sum(0).sum(0) != 0))

    def __len__(self):
      return len(self.label)
    
    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        valid_frame_num = self.valid_frame_nums[index]
        num_people = self.num_peoples[index]

        data_numpy, index_t = tools.valid_crop_uniform(data_numpy, valid_frame_num, self.p_interval, self.max_frame, 64)


        if self.train:
            # intra-instance augmentation
            p = np.random.rand()
            if p < self.intra_p:

                if 'a' in self.aug_method:
                    if np.random.rand(1) < 0.5:
                        data_numpy = data_numpy[:, :, :, np.array([1, 0])]
                if 'b' in self.aug_method:
                    if num_people == 2:
                        if np.random.rand(1) < 0.5:
                            axis_next = np.random.randint(0, 1)
                            temp = data_numpy.copy()
                            C, T, V, M = data_numpy.shape
                            x_new = np.zeros((C, T, V))
                            temp[:, :, :, axis_next] = x_new
                            data_numpy = temp

                if '1' in self.aug_method:
                    data_numpy = tools.shear(data_numpy, p=0.5)
                if '2' in self.aug_method:
                    data_numpy = tools.rotate(data_numpy, p=0.5)
                if '3' in self.aug_method:
                    data_numpy = tools.scale(data_numpy, p=0.5)
                if '4' in self.aug_method:
                    data_numpy = tools.spatial_flip(data_numpy, p=0.5)
                if '5' in self.aug_method:
                    data_numpy, index_t = tools.temporal_flip(data_numpy, index_t, p=0.5)
                if '6' in self.aug_method:
                    data_numpy = tools.gaussian_noise(data_numpy, p=0.5)
                if '7' in self.aug_method:
                    data_numpy = tools.gaussian_filter(data_numpy, p=0.5)
                if '8' in self.aug_method:
                    data_numpy = tools.drop_axis(data_numpy, p=0.5)
                if '9' in self.aug_method:
                    data_numpy = tools.drop_joint(data_numpy, p=0.5)
            # inter-instance augmentation
            
            elif (p < (self.intra_p + self.inter_p)) & (p >= self.intra_p):
                adain_idx = random.choice(np.where(self.label == label)[0])
                data_adain = self.data[adain_idx]
                data_adain = np.array(data_adain)
                f_num = np.sum(data_adain.sum(0).sum(-1).sum(-1) != 0)
                t_idx = np.round((index_t + 1) * f_num / 2).astype(np.int)
                data_adain = data_adain[:, t_idx]
                data_numpy = tools.skeleton_adain_bone_length(data_numpy, data_adain)

            else:
                data_numpy = data_numpy.copy()
        
        if self.transforms:
            data_numpy = self.transforms(data_numpy)

        else:
            data_numpy = data_numpy.copy()
        return torch.from_numpy(data_numpy.astype(np.float32)), label
    
class NPZNTUDataLoader(BaseDataLoader):
    """
    NTU RGB+D DataLoader using NPZ files
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, 
                 num_workers=64, training=True, persistent_workers=False,
                 p_interval=[0.5, 1], aug_method='a123489', intra_p=0.5, inter_p=0.0, max_frame=64):
        """
        Args:
            data_pth (str): Path to the .npz file
            batch_size (int): Batch size
            shuffle (bool): Whether to shuffle the data
            validation_split (float): Fraction of data to use as validation
            num_workers (int): Number of workers for data loading
            training (bool): Whether this is training data (affects augmentations)
            persistent_workers (bool): Whether to persist workers
            p_interval (list): Frame cropping interval
            aug_method (str): Augmentation methods to apply
            intra_p (float): Probability for intra-instance augmentation
            inter_p (float): Probability for inter-instance augmentation
            max_frame (int): Maximum number of frames
        """
        # Initialize transforms (can be empty or add necessary transforms)
        trsfm = transforms.Compose([
            # Add any required transforms here
            # Example: transforms.ToTensor() if not handled in dataset
        ])

        self.data_pth = data_dir
        self.training = training

        # Create dataset instance
        self.dataset = NPZNTURGBDDataset(
            data_pth=self.data_pth,
            train=self.training,
            transforms=trsfm,
            p_interval=p_interval,
            aug_method=aug_method,
            intra_p=intra_p,
            inter_p=inter_p,
            max_frame=max_frame
        )

        # Initialize BaseDataLoader
        super().__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            validation_split=validation_split,
            num_workers=num_workers,
            persistent_workers=persistent_workers
        )
