import pathlib

import torch
from torch.utils.data import DataLoader
import shutil
from pathlib import Path
import os
from PIL import Image
import torchvision
import xml.etree.ElementTree as ET
import numpy as np
class DogNCatDataSet(DataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir:str = None, batch_size:int = 64, transform = None):
        if not data_dir:
            data_dir = pathlib.Path.cwd().parent.joinpath('data')
        self.base_dir = pathlib.Path(data_dir)
        self.image_dir = self.base_dir.joinpath('images')
        self.annotation_dir = self.base_dir.joinpath('annotations')
        self.image_names = os.listdir(self.image_dir)
        self.batch_size = batch_size
        self.class_dict = {'dog':0, 'cat':1}
        if transform:
            self.transform = transform
        else:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((227,227)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
            ])
        print(f"total data len : {len(self.image_names)}")
    def get_image(self, file_name):
        im = Image.open(self.image_dir.joinpath(file_name)).convert('RGB')
        image = self.transform(im)
        return image
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self,idx):
        file_name = self.image_names[idx]
        class_name = os.path.join(self.annotation_dir, file_name)
        xml_class = ET.parse(class_name.replace('png','xml'))
        train_data = self.get_image(file_name)
        dog_or_cat = xml_class.find('object').findtext('name')
        y = torch.zeros(2)
        y[self.class_dict[dog_or_cat]]=1
        return train_data,y

def loader_test():
    model = DogNCatDataSet()
    print(model.__getitem__(3))
# loader_test()


def get_dataloader(name, args):
    data_loader_list = {'DogAndCat': DogNCatDataSet}
    print(args)
    print()
    return data_loader_list[name](**args)
