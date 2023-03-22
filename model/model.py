import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class DnCModel(nn.Module):
    def __init__(self, input_dim: list = [3, 224, 224], classes: int = 2, cdim:list = [12, 32, 64], k_size: int = 3, dropout : float = 0.5):
        super().__init__()
        self.input_dim = input_dim
        self.class_num = classes
        self.k_size = k_size
        self.cdims = cdim
        self.layers = []
        self.dropout = dropout

        prev_cdim = input_dim[0]

        for cdim in self.cdims:
            self.layers.append(nn.Conv2d(prev_cdim, cdim, self.k_size, padding=1))
            self.layers.append(nn.MaxPool2d(2,2))
            self.layers.append(nn.Dropout(self.dropout))
            self.layers.append(nn.ReLU())
            prev_cdim = cdim
        c_out_dim = prev_cdim * (input_dim[1] // (2 ** len(self.cdims))) * (input_dim[2] // (2 ** len(self.cdims)))
        self.layers.append(nn.Flatten(1))
        self.layers.append(nn.Linear(c_out_dim, classes))
        self.layers.append(nn.Softmax(-1))
        self.seq = nn.Sequential()

        for idx,layer in enumerate(self.layers):
            self.seq.add_module(f'layer {idx}', layer)

        self.initialize()
    def forward(self, x):
        y = self.seq(x)

        return y
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m,nn.Linear):
                nn.init.zeros_(m.bias)



def initalize_param(model:nn.Module):
    for m in model.modules():
        if isinstance(m,nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            nn.init.zeros_(m.bias)

class ConvModule(nn.Module):
    def __init__(self, input_dim, output_dim, k_size,padding:int=0, p_size=0, stride=2, dropout=0.5, use_batch=False):
        super().__init__()
        self.conv = nn.Conv2d(input_dim,output_dim,k_size,stride,padding)
        self.isPool = False
        if p_size:
            self.isPool=True
            self.pool = nn.MaxPool2d((p_size,p_size),2)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(output_dim)
        self.use_batch = use_batch
        self.initialize()
    def forward(self,x):
        #print(self.activation(self.dropout(self.conv(x))).shape)
        y=self.activation(self.dropout(self.conv(x)))
        if self.isPool:
            y = self.pool(y)
        if self.use_batch:
            y = self.batch_norm(y)
        return y
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight,0,0.01)
                nn.init.normal_(m.bias,0.5,0.3)
class AlexNet(nn.Module):
    '''
    alexnet처럼 input = 3*224*224로
    '''
    def __init__(self, classes: int = 2, k_size: int = 3, dropout: float = 0.5):
        super().__init__()
        self.Conv1_1 = ConvModule(input_dim=3,output_dim=48,k_size=11,p_size=3,stride=4,padding=0,use_batch=True)
        self.Conv1_2 = ConvModule(input_dim=3, output_dim=48, k_size=11, p_size=3, stride=4, padding=0,use_batch=True)
        self.Conv2_1 = ConvModule(input_dim=48, output_dim=128, k_size=5, p_size=3, stride=1,padding=2,use_batch=True)
        self.Conv2_2 = ConvModule(input_dim=48, output_dim=128, k_size=5, p_size=3, stride=1, padding=2,use_batch=True)
        self.Conv3_1 = ConvModule(input_dim=128, output_dim=192, k_size=3, stride=1,padding=1)
        self.Conv3_2 = ConvModule(input_dim=128, output_dim=192, k_size=3, stride=1, padding=1)
        self.Conv4_1 = ConvModule(input_dim=192, output_dim=192, k_size=3, stride=1, padding=1)
        self.Conv4_2 = ConvModule(input_dim=192, output_dim=192, k_size=3, stride=1, padding=1)
        self.Conv5_1 = ConvModule(input_dim=192,output_dim=128,k_size=3,p_size=3,stride=1,padding=1)
        self.Conv5_2 = ConvModule(input_dim=192, output_dim=128, k_size=3, p_size=3, stride=1, padding=1)
        self.Linear1 = nn.Linear(9216,4096)
        self.Linear2 = nn.Linear(4096,2)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(-1)
    def forward(self,x):
        y1 = self.Conv5_1(self.Conv4_1(self.Conv3_1(self.Conv2_1(self.Conv1_1(x)))))
        y2 = self.Conv5_2(self.Conv4_2(self.Conv3_2(self.Conv2_2(self.Conv1_2(x)))))
        concat_and_flat_y = torch.concat((torch.flatten(y1,1),torch.flatten(y2,1)),dim=1)
        y = self.activation(self.dropout(self.Linear1(concat_and_flat_y)))
        y = self.Linear2(y)
        #y = self.softmax(y)
        #print(y.shape)
        return y
