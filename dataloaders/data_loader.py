import logging

import pandas as pd
import torch
from torch.utils.data import DataLoader

from config.path import dataset_path


class GetDataLoder:

    def __init__(self, args):
        self.test_loader = None
        self.train_loader = None
        self.logger = logging.getLogger('Load_data')
        self.args = args
        self.logger.info("Load_data")
        self.way = self.__getway__(args['model'])
        self.dataset = self.getdataset(args)
        self.mode = args['mode']
        self.load()

    def getdataset(self, args):
        dataset_name = args['dataset']
        path = dataset_path(dataset_name)
        if self.way == 'normal':
            path = dataset_path(dataset_name).normal_path()
            data = pd.read_csv(path, header=0, names=['uid', 'iid', 'rating', 'timestamp'])
            return data
        elif self.way == 'seq':
            seq_len = args['seq_len']
            path = dataset_path(dataset_name).seq_path()
            data = pd.read_csv(path, header=None, names=['uid', 'iid', 'rating', 'timestamp'])
            data = self.generate_seq_data(data, seq_len)
            return data
        else:
            raise ValueError("way must be 'normal' or 'seq'")

    @staticmethod
    def generate_seq_data(data, seq_len):
        # 生成序列数据
        # todo: 生成序列数据
        return data

    def load(self):
        if self.mode == 'train':
            train_data, test_data = self.split_data_by_user_and_timestamp(self.dataset, train_ratio=0.8)
            train_dataset = CustomDataset(train_data)
            test_dataset = CustomDataset(test_data)
            self.train_loader = DataLoader(train_dataset, batch_size=self.args['batch_size'], shuffle=True)
            self.test_loader = DataLoader(test_dataset, batch_size=self.args['batch_size'], shuffle=True)
        if self.mode == 'if':
            pass

    @staticmethod
    def split_data_by_user_and_timestamp(data, train_ratio=0.8):
        # Assuming 'data' is a pandas DataFrame with columns ['uid', 'iid', 'rating', 'timestamp']
        train_data = pd.DataFrame(columns=data.columns)
        test_data = pd.DataFrame(columns=data.columns)

        for uid in data['uid'].unique():
            user_data = data[data['uid'] == uid].sort_values(by='timestamp')
            split_index = int(len(user_data) * train_ratio)

            user_train_data = user_data[:split_index]
            user_test_data = user_data[split_index:]

            train_data = pd.concat([train_data, user_train_data])
            test_data = pd.concat([test_data, user_test_data])

        return train_data, test_data

    @staticmethod
    def __getway__(model_name):
        print(model_name)
        if model_name in ['GRU4Rec', 'SASRec']:
            return 'seq'
        else:
            return 'normal'

    # def load_test(self):
    #     if self.dataset == 'mnist':
    #         test_loader = torch.utils.data.DataLoader(
    #             datasets.MNIST('dataset', train=False, transform=transforms.Compose([
    #                 transforms.ToTensor(),
    #                 transforms.Normalize((0.1307,), (0.3081,))
    #             ])),
    #             batch_size=self.args['batch_size'], shuffle=True)
    #         return test_loader
    #
    # def load_unlearn(self, train_dataset):
    #     if self.unlearn_data == 'random':
    #         unlearn_dataset = torch.utils.data.SubsetRandomSampler(
    #             torch.randperm(len(train_dataset))[:int(len(train_dataset) * self.unlearn_ratio)])
    #         unlearn_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args['batch_size'],
    #                                                      sampler=unlearn_dataset)
    #         return unlearn_loader


# Assuming 'train_data' is a pandas DataFrame and you've already defined a dataset class that can handle this DataFrame
# Let's also assume your dataset class is named CustomDataset and it takes a pandas DataFrame as input

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        # Initialize dataset, e.g., convert pandas DataFrame to tensors
        self.data = data

    def __len__(self):
        # Return the size of the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Implement logic to get a single item at index `idx`
        return self.data.iloc[idx]
