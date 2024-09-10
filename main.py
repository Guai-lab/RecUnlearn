import logging
import random

import numpy as np
import torch

from exp.exp_if import IF
from exp.exp_train import Train
from parameter_parser import parameter_parser


def _set_random_seed(seed=2023):
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    print("set pytorch seed")


def _set_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s:%(asctime)s: - %(name)s - : %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def main():
    _set_random_seed()
    _set_logger()
    args = parameter_parser()
    logging.info(args)

    if args['mode'] == 'train':
        Train(args)
    elif args['mode'] == 'if':
        IF(args)


if __name__ == '__main__':
    main()
