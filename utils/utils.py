import numpy as np
import warnings
import os
from sklearn.preprocessing import minmax_scale
import random
import logging
from logging import handlers
import torch.multiprocessing
import torch

warnings.filterwarnings("ignore")

def load_lt_data(dataset_name, split):
    data = np.load(f'{os.path.dirname(__file__)}/../data/dataset/{dataset_name}.npz')
    x, y = data['x'], data['y']
    x = minmax_scale(x)
    if split is None:
        return x, y
    idx = np.load(f'{os.path.dirname(__file__)}/../data/lt_idx/{dataset_name}_lt_idx.npy', allow_pickle=True)[split]
    train_idx, test_idx = idx['train_idx'].squeeze(), idx['test_idx'].squeeze()
    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]
    return x_train, y_train, x_test, y_test


def set_seed(seed):
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    torch.cuda.manual_seed_all(seed)  # multi-gpu
    torch.backends.cudnn.deterministic = True  # cudnn
    torch.backends.cudnn.benchmark = False  # cudnn
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    def __init__(self, filename, level='info', when='D', backCount=3,
                 fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))
        sh = logging.StreamHandler()
        th = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=backCount, encoding='utf-8')
        th.setFormatter(format_str)
        self.logger.addHandler(sh)
        self.logger.addHandler(th)
