import logging

from exp.base_exp import Exp


class Train(Exp):
    def __init__(self, args):
        super().__init__(args)
        self.logger = logging.getLogger('Train')
        self.logger.info("Train")
        self.model = self.load_model().to(self.device)
        self.epoch = self.args['epochs']
        self.train_data = self.dataloder.train_loader
        self.test_data = self.dataloder.test_loader
        self.run()
        self.save_point(self.model)

    def run(self):
        for epoch in range(1, self.epoch + 1):
            self.logger.info("epoch: {}".format(epoch))
            self.train(self.model, self.device, self.train_data, epoch)
            self.test(self.model, self.device, self.test_data)
