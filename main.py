import logging
import torch
import numpy as np
import random
from parameter_parser import parameter_parser
from exp.exp_train import Train


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
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def main():
    args = parameter_parser()
    _set_random_seed()
    _set_logger()
    if args['mode'] == 'train':
        Train(args)


if __name__ == '__main__':
    main()
