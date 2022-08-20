import torch
from torch import nn
from torch.nn import functional as F

import numpy as np


class CAModel(nn.Module):
    def __init__(self, hyperparams):
        super(CAModel, self).__init__()

        self.device = hyperparams['Device']
        self.batch_sz = hyperparams['Batch Size']
        self.hidden_size = hyperparams['Hidden dim.']
        self.channel_n = hyperparams['Channels']
        self.laplace = hyperparams['Laplace']
        self.fire_rate = hyperparams['Fire rate']
        self.model_type = hyperparams['Model Type']

        self.to(self.device)

        if self.model_type == 'FC':
            self.layer_0 = nn.Linear(self.channel_n * (3 + self.laplace), self.hidden_size)
            self.layer_1 = nn.Linear(self.hidden_size, self.channel_n, bias=False)
            with torch.no_grad():
                self.layer_1.weight.zero_()

            self.propogate = self.propogate_fc

        else:
            self.layer_0 = nn.Conv2d(self.channel_n* (3 + self.laplace),
                                     self.hidden_size, 3, 1, padding=1)
            self.layer_1 = nn.Conv2d(self.hidden_size, self.channel_n, 1, 1)

            nn.init.zeros_(self.layer_1.weight)
            nn.init.zeros_(self.layer_1.bias)

            self.propogate = self.propogate_cnn

        self.max_pool = nn.MaxPool2d(3, 1, padding=1)
        self.relu = nn.ReLU()
        self.sobel_x = None
        self.sobel_y = None

        self.prep_deriviatives()

    def prep_deriviatives(self):
        self.sobel_x = torch.from_numpy(np.outer([1, 2, 1],
                                                 [-1, 0, 1]) / 8.0).float()
        self.sobel_y = self.sobel_x.T

        self.sobel_x = self.sobel_x.repeat(self.channel_n, 1, 1, 1).to(self.device)
        self.sobel_y = self.sobel_y.repeat(self.channel_n, 1, 1, 1).to(self.device)

        self.laplas = torch.tensor([[1.0, 2.0, 1.0],
                                    [2.0, -12, 2.0],
                                    [1.0, 2.0, 1.0]])
        self.laplas = self.laplas.repeat(self.channel_n, 1, 1, 1).to(self.device)

    # Steps
    def alive(self, x):
        return self.max_pool(x[:, 3, :, :]) > 0.1

    def perceive(self, x, angle, laplace=False):

        c = np.cos(angle * np.pi / 180)
        s = np.sin(angle * np.pi / 180)
        w1 = c * self.sobel_x - s * self.sobel_y
        w2 = s * self.sobel_x + c * self.sobel_y

        y1 = F.conv2d(x, w1, padding=1, groups=self.channel_n)
        y2 = F.conv2d(x, w2, padding=1, groups=self.channel_n)

        y = torch.cat((x, y1, y2), 1)

        if laplace:
            y3 = F.conv2d(x, self.laplas, padding=1, groups=self.channel_n)
            y = torch.cat((y, y3), 1)

        return y

    def update(self, x, fire_rate, angle, laplace):
        pre_life_mask = self.alive(x)

        delta = self.perceive(x, angle, laplace)

        delta = self.propogate(delta)

        if fire_rate is None:
            fire_rate = self.fire_rate

        stochastic = torch.rand([delta.size(0), 1, delta.size()[-2], delta.size()[-1]]) > fire_rate
        stochastic = stochastic.float().to(self.device)
        delta = delta * stochastic

        x = x + delta

        post_life_mask = self.alive(x)
        life_mask = (pre_life_mask & post_life_mask).float()
        x = x * life_mask.unsqueeze(1)
        return x

    def propogate_cnn(self, x):
        x = self.layer_0(x)
        x = self.relu(x)
        x = self.layer_1(x)

        return x

    def propogate_fc(self,x):

        x = x.transpose(1, 3)

        x = self.layer_0(x)
        x = self.relu(x)
        x = self.layer_1(x)

        x = x.transpose(1, 3)

        return x

    def forward(self, x, steps=1, fire_rate=None, angle=0.0):
        for step in range(steps):
            x = self.update(x, fire_rate, angle, self.laplace)
        return x
