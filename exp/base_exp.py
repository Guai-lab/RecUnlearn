import logging
import os

import torch
import torch.optim as optim

from dataloaders.data_loader import GetDataLoder
from models.convnet import ConvNet
from models.mf import MF
from models.mlp import MLP


class Exp:
    def __init__(self, args):
        self.logger = logging.getLogger('Exp')
        self.dataloder = None
        self.device = None
        self.args = args
        self.init()

    # 参数初始化
    def init(self):
        self.device = torch.device(self.args['cuda'] if torch.cuda.is_available() else 'cpu')
        self.dataloder = GetDataLoder(self.args)

    def load_model(self):
        if self.args['model'] == 'ConvNet':
            return ConvNet()
        elif self.args['model'] == 'MLP':
            return MLP()
        elif self.args['model'] == 'mf':
            return MF(self.dataloder.num_users, self.dataloder.num_items, 50)

    def train(self, model, device, train_loader, epoch):
        self.logger.info("training")
        optimizer = optim.Adam(model.parameters(), weight_decay=self.args['l2_lambda'])
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            data, target = batch
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            # 将损失都放置在model中，以应对回归/分类等不同的任务
            # loss = F.nll_loss(output, target)
            loss = model.getloss(output, target)
            l2_reg = torch.tensor(0.).to(device)
            for param in model.parameters():
                l2_reg += torch.norm(param) ** 2
            loss += self.args['l2_lambda'] * l2_reg * len(data)
            loss.backward()
            optimizer.step()
            if (batch_idx + 1) % 200 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

    def test(self, model, device, test_loader):
        self.logger.info("testing")
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)

                # TODO 具体的指标运算需要创建在单独的文件夹中
                # 输入为output和target，输出为具体的指标值

                # test_loss += F.nll_loss(output, target, reduction='sum').item()  # 将一批的损失相加
                # pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
                # correct += pred.eq(target.view_as(pred)).sum().item()

        # avg_loss = test_loss / len(test_loader.dataset)
        # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #     avg_loss, correct, len(test_loader.dataset),
        #     100. * correct / len(test_loader.dataset)))
        return test_loss

    def save_point(self, model):
        # TODO 更加合理的保存模型方式

        path = 'checkpoint' + '/' + str(self.args['dataset'])
        file = str(self.args['model']) + '_' + str(self.args['mode']) + '_l2' + str(self.args['l2_lambda']) + '.pt'
        self.logger.info("saving model to {}".format(path))
        if not os.path.exists(path):
            self.logger.info("creating checkpoint dir")
            os.mkdir(path)
        torch.save(model.state_dict(), path + '/' + file)

    def load_point(self):
        path = 'checkpoint' + '/' + str(self.args['dataset'])
        file = str(self.args['model']) + '_' + 'train' + '_l2' + str(self.args['l2_lambda']) + '.pt'
        if not os.path.exists(path + '/' + file):
            self.logger.info("no checkpoint found at {}".format(path))
            return None
        self.logger.info("loading model from {}".format(path + '/' + file))
        model = self.load_model()
        model.load_state_dict(torch.load(path + '/' + file))
        return model
