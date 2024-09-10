import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        out = self.fc1(x.view(-1, 28 * 28))
        out = self.fc2(out)
        out = F.log_softmax(out, dim=1)
        return out
