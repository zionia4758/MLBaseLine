import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from pathlib import Path
from abc import *


class BaseDataLoader(DataLoader, metaclass=ABCMeta):
    """
    base dataloaer model \n
    상속받아 사용할 시 super().__init__()호출\n
    필수구현\n
    __init__()\n
    __getitem__()\n
    __getData()\n
    __getAanotation()\n
    """

    def __init__(self, base_dir, data_dir, annotation_dir, shuffle=None,
                 validation_split=None, transform=None):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir.joinpath(data_dir)
        self.annotation_dir = self.base_dir.joinpath(annotation_dir)
        self.data_list = self.getData(self.data_dir)
        self.annotation_list = self.getAnnotation(self.annotation_dir)
        self.transform = transform
        self.shuffle = shuffle
        self.valid_split = validation_split

    def __getitem__(self, idx):
        raise NotImplementedError

    @abstractmethod
    def getData(self, data_dir):
        raise NotImplementedError

    @abstractmethod
    def getAnnotation(self, annotation_dir):
        raise NotImplementedError
