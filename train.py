import argparse
import collections
import torch
import numpy as np
from parse_config import ConfigParser
from utils import prepare_device
from data_loader import data_loaders
import json

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
np.random.seed(SEED)

def main(config):
    SEED = config['seed']
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = config['name']
    data_loader_cfg = config['data_loader']
    print(data_loader_cfg['type'])
    print(data_loader_cfg['args'])
    data_loader = data_loaders.get_dataloader(data_loader_cfg['type'], data_loader_cfg['args'])

    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='templete project')
    parser.add_argument('-c', '--config', type=str, default="config.json",
                      help='config file path (default: ./config.json')
    parser.add_argument('-m', '--multimodel', type=str, default='',
                        help='if train multiple models')
    args = parser.parse_args()
    print(f'config file path: args.config')
    with open(args.config, 'r') as f:
        config = json.load(f)
    #config = ConfigParser.from_args(args, options)
    main(config)
