import torch.utils.data

from base.base_data_loader import BaseDataLoader
from pathlib import Path
class testDataset(BaseDataLoader):
    def __init__(self,base_dir,data,annotation,shuffle=None,
                 valid=0.9,trans=None):
        super().__init__('../data','images','annotation')
        print("test")

    def getData(self,data_dir):
        data_list = list(data_dir.iterdir())
        print(data_list[:5])
        return data_list
    def getAnnotation(self,annotation_dir):
        annotation_list = annotation_dir
        print("getanno",annotation_dir)
        return annotation_list

    def __len__(self):
        return len(self.data_list)
    def __getitem__(self,idx):
        return self.data_list[idx]


# test = testDataset("1","2","3")
# print(len(test))
# train,valid = torch.utils.data.random_split(test,[0.99,0.01])
# print(len(train),len(valid))
# for i in iter(valid):
#     print(i)