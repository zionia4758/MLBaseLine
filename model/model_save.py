import torch
import pathlib
import os

DEFAULT_DIR = '../saved_models/'


def save_model(model, epoch, accuracy, dir=None):
    if not dir:
        path = DEFAULT_DIR
    else:
        path = dir
    torch.save(model.state_dict(), os.path.join(path, f'{model._get_name()}_epoch_{epoch}_acc_{accuracy:.3f}'))
    print(f'model saved: {model._get_name()}_epoch_{epoch}_acc_{accuracy:.3f}')
