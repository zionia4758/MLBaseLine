import torch
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, criterion, metric_f, optimizer, config):
        self.name = model
        self.criterion = criterion
        self.metric = metric_f
        self.optimizaer = optimizer
        self.config = config
