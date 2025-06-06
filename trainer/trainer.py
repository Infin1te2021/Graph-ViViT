import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, EdgeIndexGenerator

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class Trainer(BaseTrainer):
  """
  Trainer class
  """
  def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
              data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
    super().__init__(model, criterion, metric_ftns, optimizer, config)
    self.config = config
    self.device = device
    self.data_loader = data_loader
    if len_epoch is None:
      # epoch-based training
      self.len_epoch = len(self.data_loader)
    else:
      # iteration-based training
      self.data_loader = inf_loop(data_loader)
      self.len_epoch = len_epoch
    self.valid_data_loader = valid_data_loader
    self.do_validation = self.valid_data_loader is not None
    self.lr_scheduler = lr_scheduler
    self.log_step = int(np.sqrt(data_loader.batch_size))

    self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
    self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    self.edge_index = EdgeIndexGenerator().generate_edge_index().to(self.device)

  def _train_epoch(self, epoch):
    """
    Training logic for an epoch

    :param epoch: Integer, current training epoch.
    :return: A log that contains average loss and metric in this epoch.
    """
    self.model.train()
    self.train_metrics.reset()
    

    for batch_idx, (data, target) in enumerate(self.data_loader):
      current_step = (epoch - 1) * self.len_epoch + batch_idx

      if self.lr_scheduler is not None and not self.lr_scheduler.t_in_epochs:
        self.lr_scheduler.step_update(current_step)
      elif isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        pass
      else:
        pass
      
      data, target = data.to(self.device), target.to(self.device)
      output = self.model(data)
      loss = self.criterion(output, target)

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      current_lr = self.optimizer.param_groups[0]['lr']
      
      self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
      self.writer.add_scalar('learning_rate', current_lr)

      self.train_metrics.update('loss', loss.item())
      for met in self.metric_ftns:
        self.train_metrics.update(met.__name__, met(output, target))

      if batch_idx % self.log_step == 0:
        self.logger.debug('Train Epoch: {} {} Loss: {:.6f} LR: {:.6f}'.format(
          epoch,
          self._progress(batch_idx),
          loss.item(),
          current_lr))

      if batch_idx == self.len_epoch:
        break
    log = self.train_metrics.result()

    if self.do_validation:
      val_log = self._valid_epoch(epoch)
      log.update(**{'val_'+k : v for k, v in val_log.items()})

    if self.lr_scheduler is not None:
      if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        # Pass validation loss to ReduceLROnPlateau
        val_loss = log.get('val_loss', None)
        if val_loss is not None:
          self.lr_scheduler.step(val_loss)

      ## CosineLRScheduler update according to epoch (Not test yet)
      elif self.lr_scheduler.t_in_epochs:
        self.lr_scheduler.step()
      else:
        if self.config["lr_scheduler"]["type"] != 'CosineLRScheduler':
          self.lr_scheduler.step()

    return log

  def _valid_epoch(self, epoch):
    """
    Validate after training an epoch

    :param epoch: Integer, current training epoch.
    :return: A log that contains information about validation
    """
    self.model.eval()
    self.valid_metrics.reset()
    all_preds = []
    all_targets = []

    with torch.no_grad():
      for batch_idx, (data, target) in enumerate(self.valid_data_loader):
        data, target = data.to(self.device), target.to(self.device)
        output = self.model(data)
        loss = self.criterion(output, target)

        preds = output.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(target.cpu())
        
        self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
        self.valid_metrics.update('loss', loss.item())
        for met in self.metric_ftns:
          self.valid_metrics.update(met.__name__, met(output, target))
        
    
      all_preds = torch.cat(all_preds).numpy()
      all_targets = torch.cat(all_targets).numpy()
      cm = confusion_matrix(all_targets, all_preds)

      figure = plt.figure(figsize=(25.6, 14.4), dpi=100)
      sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
      plt.xlabel('Predicted Labels')
      plt.ylabel('True Labels')
      plt.title(f'Confusion Matrix (Epoch {epoch})')
      plt.tight_layout()

      self.writer.add_figure('confusion_matrix/valid', figure)
      plt.close(figure)


    # add histogram of model parameters to the tensorboard
    for name, p in self.model.named_parameters():
      self.writer.add_histogram(name, p, bins='auto')
    return self.valid_metrics.result()
  

  def _progress(self, batch_idx):
    base = '[{}/{} ({:.0f}%)]'
    if hasattr(self.data_loader, 'n_samples'):
      current = batch_idx * self.data_loader.batch_size
      total = self.data_loader.n_samples
    else:
      current = batch_idx
      total = self.len_epoch
    return base.format(current, total, 100.0 * current / total)