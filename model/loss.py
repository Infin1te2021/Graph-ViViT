import torch
import torch.nn as nn

def CrossEntropyLoss(output, target):
  loss_fn = nn.CrossEntropyLoss()
  loss = loss_fn(output, target)
  return loss