import logging

import torch
import torch.nn.functional as F
from torch.autograd import grad

from exp.base_exp import Exp


class IF(Exp):
    def __init__(self, args):
        super().__init__(args)
        self.logger = logging.getLogger('IF')
        self.logger.info("IF")
        self.args = args
        self.model = self.load_point().to(self.device)
        self.test(self.model, self.device, self.test_data)
        self.unlearn_model = self.model
        self.example_data = self.train_data
        self.influence_function_par()
        self.test(self.unlearn_model, self.device, self.test_data)

    def get_loss(self, dataset):
        loss = 0
        for batch_idx, (data, target) in enumerate(dataset):
            l2_reg = torch.tensor(0.).to(self.device)
            data, target = data.to(self.device), target.to(self.device)
            output = self.unlearn_model(data)
            loss += F.nll_loss(output, target, reduction='sum')
            for param in self.unlearn_model.parameters():
                l2_reg += torch.norm(param) ** 2
            loss += self.args['l2_lambda'] * l2_reg * len(data)
        return loss

    def influence_function_par(self):
        model_params = [p for p in self.model.parameters() if p.requires_grad]
        num_params = sum(p.numel() for p in model_params)

        loss_unlearn = self.get_loss(self.unlearn_data)
        grad_unlearn = grad(loss_unlearn, model_params, retain_graph=True, create_graph=True)
        # 1*p(p为参数的个数)
        h = grad_unlearn
        example = self.example_data

        # scale = len(example.sampler) * num_params / float(self.args['scale_num'])
        scale = float(self.args['scale_num'])
        print('scale:', scale)

        loss_example = self.get_loss(self.example_data)
        grad_example = grad(loss_example, model_params, create_graph=True)
        for i in range(int(self.args['iteration'])):
            hv = 0
            for grad_elem, v_elem in zip(grad_example, h):
                hv += torch.sum(grad_elem * v_elem)
            second_order = grad(hv, model_params, create_graph=True)
            if i % 10 == 0:
                print(hv)
            with torch.no_grad():
                h = [gradi + hi - hvi / scale for gradi, hi, hvi in zip(grad_unlearn, h, second_order)]
        params_change = [hi / scale for hi in h]
        print(params_change)

        # for _ in range(100):
        # # for example in self.example_data:
        #     hv = self.second_order(model_params, example, h)
        #     with torch.no_grad():
        #         h = [gradi+hi-hvi/len(example.sampler) for gradi, hi, hvi in zip(grad_unlearn, h, hv)]

        # params_change = [influence / len(self.train_data) for influence in h]
        unlearn_params = [p1 + p2 for p1, p2 in zip(model_params, params_change)]
        i, temp = 0, self.unlearn_model.parameters()
        temp = unlearn_params
        for p in self.unlearn_model.parameters():
            p.data = unlearn_params[i]
            i += 1
        print('判断是否相等', temp == self.unlearn_model.parameters())
        self.save_point(self.unlearn_model)

    def second_order(self, model_params, example, h):
        # example = [example]
        hv = 0
        loss_example = self.get_loss(example)
        grad_example = grad(loss_example, model_params, create_graph=True)

        # hv = 0
        # grad_example = example

        for grad_elem, v_elem in zip(grad_example, h):
            hv += torch.sum(grad_elem * v_elem)
        second_order = grad(hv, model_params)
        return second_order

    def influence_function_loss(self):
        pass
