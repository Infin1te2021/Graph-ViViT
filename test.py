import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path

def main(config):
  logger = config.get_logger('test')

  # setup data_loader instances
  data_loader = getattr(module_data, config['data_loader']['type'])(
    config['data_loader']['args']['data_dir'],
    batch_size=128,
    shuffle=False,
    validation_split=0.0,
    training=False,
    num_workers=64
  )

  # build model architecture
  model = config.init_obj('arch', module_arch)
  logger.info(model)

  # get function handles of loss and metrics
  loss_config = config['loss']
  criterion_type = loss_config['type']
  criterion_args = loss_config.get('args', {})

  # Dynamically initialize the loss function
  try:
    # Fetch the loss class dynamically from torch.nn
    criterion_cls = getattr(nn, criterion_type)
    # Initialize the loss function with provided arguments
    loss_fn = criterion_cls(**criterion_args)
  except AttributeError:
    raise ValueError(f"Loss function {criterion_type} not found in torch.nn.")
  except TypeError as e:
    raise ValueError(f"Error initializing loss function {criterion_type}: {e}")
  metric_fns = [getattr(module_metric, met) for met in config['metrics']]

  logger.info('Loading checkpoint: {} ...'.format(config.resume))
  checkpoint = torch.load(config.resume)

  if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    raw_weights = checkpoint['state_dict']
    print('Load weights from dict')
  elif isinstance(checkpoint, torch.nn.Module):
    raw_weights = checkpoint.state_dict()
    print('Load weights from model')
  else:
    raw_weights = checkpoint
    print('Load weights from directly')

  if config['n_gpu'] > 1:
    model = torch.nn.DataParallel(model)
  model.load_state_dict(raw_weights)

  # prepare model for testing
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = model.to(device)
  model.eval()

  total_loss = 0.0
  total_metrics = torch.zeros(len(metric_fns))


  all_preds = []
  all_targets = []
  
  with torch.no_grad():
    for i, (data, target) in enumerate(tqdm(data_loader)):
      data, target = data.to(device), target.to(device)
      output = model(data)
      #
      # save sample images, or do something with output here
      #
      preds = output.argmax(dim=1)
      all_preds.append(preds.cpu())
      all_targets.append(target.cpu())

      # computing loss, metrics on test set
      loss = loss_fn(output, target)
      batch_size = data.shape[0]
      total_loss += loss.item() * batch_size
      for i, metric in enumerate(metric_fns):
        total_metrics[i] += metric(output, target) * batch_size

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    cm = confusion_matrix(all_targets, all_preds)

    model_dir = Path(config.resume).parent.parent.name
    parts = model_dir.split('_')

    test_code = parts[2] if len(parts) > 2 else 'Unknown'

    match = re.match(r'(\d+)([A-Za-z]+)', test_code)
    dataset, code = match.groups() if match else ('Unknown', 'Unknown')

    test_type_mapping = {
      '60CV': 'Cross View',
      '60CS': 'Cross Subject',
      '120CSet': 'Cross Set',
      '120CSub': 'Cross Subject'
    }
    test_name = test_type_mapping.get(code, 'Unknown Test')

    figure = plt.figure(figsize=(21.6, 21.6), dpi=100)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={
      'size': 8, 
    })
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'VideoViT: NTU-RGB+D {dataset} {test_name} Test Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'video_vit_{dataset}_{code.lower()}_CM.pdf', format='pdf', bbox_inches='tight')
    
  n_samples = len(data_loader.sampler)
  log = {'loss': total_loss / n_samples}
  log.update({
    met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
  })
  logger.info(log)


if __name__ == '__main__':
  args = argparse.ArgumentParser(description='PyTorch Template')
  args.add_argument('-c', '--config', default=None, type=str,
                    help='config file path (default: None)')
  args.add_argument('-r', '--resume', default=None, type=str,
                    help='path to latest checkpoint (default: None)')
  args.add_argument('-d', '--device', default=None, type=str,
                    help='indices of GPUs to enable (default: all)')

  config = ConfigParser.from_args(args)
  main(config)
