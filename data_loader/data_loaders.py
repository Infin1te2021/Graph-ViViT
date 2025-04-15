import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from base import BaseDataLoader
import numpy as np
import torch.nn.functional as F
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
