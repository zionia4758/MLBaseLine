import torch.nn.functional as F
import torch.nn as nn

def nll_loss(output, target):
    return F.nll_loss(output, target)
def set_loss(loss_f : str):
    loss_functions = ['MSE','BCE', "CE"]
    if not loss_f:
        return nn.BCELoss()
    if loss_f not in loss_functions:
        raise AttributeError("구현되지 않았거나 잘못된 loss function을 입력했습니다.")
    if loss_f == 'BCE':
        print('BCE Loss')
        criterion = nn.BCELoss()
    elif loss_f == 'MSE':
        print('MSE Loss')
        criterion = nn.MSELoss()
    elif loss_f == 'CE':
        print('CE Loss')
        criterion = nn.CrossEntropyLoss()
    return criterion