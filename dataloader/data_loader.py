import logging
from typing import Tuple, Any

import torch
from torchvision import datasets, transforms


class Load_data:

    def __init__(self, args):
        self.logger = logging.getLogger('Load_data')
        self.args = args
        self.logger.info("Load_data")
        self.dataset = args['dataset']
        self.mode = args['mode']
        self.data = self.load()

    def load(self):
        datadir = './dataset' + '/' + self.dataset
        if self.mode == 'train':
            train_loader = self.load_train()
            test_loader = self.load_test()
            return train_loader, None, test_loader

    def load_train(self):
        if self.dataset == 'mnist':
            train_loader = torch.utils.data.DataLoader(
                datasets.MNIST('dataset', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ])),
                batch_size=self.args['batch_size'], shuffle=True)
            return train_loader

    def load_test(self):
        if self.dataset == 'mnist':
            test_loader = torch.utils.data.DataLoader(
                datasets.MNIST('dataset', train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])),
                batch_size=self.args['batch_size'], shuffle=True)
            return test_loader
