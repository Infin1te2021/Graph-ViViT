import argparse
import collections
import torch
import torch.nn as nn
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
from utils import GradualWarmupScheduler
from timm.scheduler.cosine_lr import CosineLRScheduler

# fix random seeds for reproducibility
SEED = 1
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
  logger = config.get_logger('train')

  # setup data_loader instances
  data_loader = config.init_obj('data_loader', module_data)

  valid_data_loader = data_loader.split_validation()

  # build model architecture, then print to console
  model = config.init_obj('arch', module_arch)
  if config['pretrain']:  # 建议通过配置文件指定路径
    logger.info(f"Loading pretrained model from {config['pretrain']}")
    checkpoint = torch.load(config['pretrain'])
    
    # 处理可能的DataParallel前缀
    state_dict = checkpoint
    if list(state_dict.keys())[0].startswith('module.'):
      state_dict = {k[7:]: v for k, v in state_dict.items()}  # 去除module.前缀
    
    load_status = model.load_state_dict(state_dict, strict=False)
    logger.info(f"Missing keys: {load_status.missing_keys}")  # 显示缺失的键（新层）
    logger.info(f"Unexpected keys: {load_status.unexpected_keys}")  # 显示多余的键（旧分类层）
  
  logger.info("Freezing original model parameters")

  # freeze_patterns = [
  #   # 'st_gcn_networks',  # 主网络层
  #   # 'data_bn',          # 数据批归一化层 
  #   # 'edge_importance'   # 边权重参数
  # ]
  # for name, param in model.named_parameters():
  #   if any(pattern in name for pattern in freeze_patterns):
  #       param.requires_grad = False
  #       logger.debug(f"Frozen parameter layers ❄️: {name}")
  #   # 保持新增层可训练
  #   elif 'adapt' in name or 'vit' in name:
  #       param.requires_grad = True
  #       logger.debug(f"Trainable layers 🔥: {name}")
  #   # 其他未明确指定的层默认保持原状（建议明确处理）
  #   else:
  #       logger.warning(f"Unhandled: {name} Current Status:{'🔥' if param.requires_grad else '❄️'}")

  logger.info(model)
  # prepare for (multi-device) GPU training
  device, device_ids = prepare_device(config['n_gpu'])
  model = model.to(device)
  if len(device_ids) > 1:
    model = torch.nn.DataParallel(model, device_ids=device_ids)

  trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
  
  loss_config = config['loss']
  criterion_type = loss_config['type']
  criterion_args = loss_config.get('args', {})

  # Dynamically initialize the loss function
  try:
    # Fetch the loss class dynamically from torch.nn
    criterion_cls = getattr(nn, criterion_type)
    # Initialize the loss function with provided arguments
    criterion = criterion_cls(**criterion_args)
  except AttributeError:
    raise ValueError(f"Loss function {criterion_type} not found in torch.nn.")
  except TypeError as e:
    raise ValueError(f"Error initializing loss function {criterion_type}: {e}")

  metrics = [getattr(module_metric, met) for met in config['metrics']]

  trainable_params = filter(lambda p: p.requires_grad, model.parameters())
  optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

  lr_scheduler = None
  if 'lr_scheduler' in config.config:
    scheduler_config = config.config['lr_scheduler']
    scheduler_type = scheduler_config['type']
    if scheduler_type.strip():
      if scheduler_type == 'GradualWarmupScheduler':
        total_epoch = scheduler_config['args']['warm_up_epoch']
        after_scheduler_config = scheduler_config['args']['after_scheduler']
        
        after_scheduler_type = after_scheduler_config['type']
        after_scheduler_args = after_scheduler_config.get('args', {})
        after_scheduler_cls = getattr(torch.optim.lr_scheduler, after_scheduler_type)
        after_scheduler = after_scheduler_cls(optimizer, **after_scheduler_args)

        lr_scheduler = GradualWarmupScheduler(
          optimizer, 
          total_epoch=total_epoch, 
          after_scheduler=after_scheduler
        )
      if scheduler_type == 'CosineLRScheduler':
        print("Using timm CosineLRScheduler")

        num_steps = int(len(data_loader)  * config['trainer']['epochs'])
        
        warmup_steps = int(len(data_loader) * config["lr_scheduler"]["args"]["warm_up_epoch"])
        t_initial = (num_steps - warmup_steps) if scheduler_config['args']['warmup_prefix'] else num_steps

        lr_scheduler = CosineLRScheduler(optimizer, 
                                        t_initial=t_initial,
                                        lr_min = scheduler_config['args']['lr_min'],
                                        warmup_lr_init = scheduler_config['args']['warmup_lr_init'],
                                        warmup_t=warmup_steps,
                                        warmup_prefix= scheduler_config['args']['warmup_prefix'],
                                        cycle_limit=1,
                                        t_in_epochs=scheduler_config['args']['t_in_epochs'])
      else:
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
  else:
    lr_scheduler = None
  print(f"Scheduler Params: {lr_scheduler.__dict__}")
  # setup trainer
  trainer = Trainer(model, criterion, metrics, optimizer,
                    config=config,
                    device=device,
                    data_loader=data_loader,
                    valid_data_loader=valid_data_loader,
                    lr_scheduler=lr_scheduler)

  return trainer.train()


if __name__ == '__main__':
  args = argparse.ArgumentParser(description='Deep Learning for Skeleton-based Action Recognition')
  args.add_argument('-c', '--config', default=None, type=str,
                    help='config file path (default: None)')
  args.add_argument('-r', '--resume', default=None, type=str,
                    help='path to latest checkpoint (default: None)')
  args.add_argument('-d', '--device', default=None, type=str,
                    help='indices of GPUs to enable (default: all)')

  # custom cli options to modify configuration from default values given in json file.
  CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
  options = [
    CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
    CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
  ]
  config = ConfigParser.from_args(args, options)
  main(config)