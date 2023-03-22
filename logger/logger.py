import logging
import logging.config
import os.path
import pathlib
from pathlib import Path
from utils import read_json
from torch.utils.tensorboard import SummaryWriter


def tensorboard_logger(log_dir: str, file_name: str):
    if os.path.isdir(log_dir):
        writer = SummaryWriter(pathlib.Path.joinpath(log_dir, file_name))
    return writer


def tensorboard_write(writer, epoch, data):

    print(data)
    for key in data:
        if key == 'sampleOutput':
            writer.add_embedding(key,data[key],epoch)
            continue
        writer.add_scalar(key, data[key], epoch)
    writer.flush()


def setup_logging(save_dir, log_config='logger/logger_config.json', default_level=logging.INFO):
    """
    Setup logging configuration
    """
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        # modify logging paths based on run config
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(save_dir / handler['filename'])

        logging.config.dictConfig(config)
    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)
