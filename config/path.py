import os
from enum import Enum

root_path = '/home/lcwt/guos/experiment/RecUnlearn-torch'


class dataset(Enum):
    ml1m = root_path + '/datasets/ml1m'


class dataset_path:
    def __init__(self, dataset_name):
        self.path = dataset[dataset_name].value

    def normal_path(self):
        return self.path + '/ratings' if os.path.exists(self.path + '/ratings') else None

    def seq_path(self):
        return self.path + '/ratings' if os.path.exists(self.path + '/ratings') else None
