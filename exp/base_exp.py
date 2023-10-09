import os
import torch
import logging
import torch.optim as optim
import torch.nn.functional as F
from dataloader.data_loader import Load_data
from model import ConvNet


class Exp:
    def __init__(self, args):
        self.logger = logging.getLogger('Exp')
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = Load_data(args)
        self.train_data, self.unlearn_data, self.test_data = data.data

    def load_model(self):
        if self.args['model'] == 'ConvNet':
            return ConvNet.ConvNet()

    def train(self, model, device, train_loader, epoch):
        self.logger.info("training")
        optimizer = optim.Adam(model.parameters())
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
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
                test_loss += F.nll_loss(output, target, reduction='sum').item() # 将一批的损失相加
                pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    def save_point(self, model):
        path = 'checkpoint' + '/' + str(self.args['dataset']) + '/' + str(self.args['model']) + '_' + str(self.args['mode']) + '.pt'
        self.logger.info("saving model to {}".format(path))
        if not os.path.exists('checkpoint' + '/' + str(self.args['dataset'])):
            self.logger.info("creating checkpoint dir")
            os.mkdir('checkpoint' + '/' + str(self.args['dataset']))
        torch.save(model.state_dict(), path)

    def load_point(self):
        path = 'checkpoint' + '/' + str(self.args['dataset']) + '/' + str(self.args['model']) + '_' + str(self.args['mode']) + '.pt'
        if not os.path.exists(path):
            self.logger.info("no checkpoint found at {}".format(path))
            return
        self.logger.info("loading model from {}".format(path))
        model = self.load_model()
        model.load_state_dict(torch.load(path))
        return model
