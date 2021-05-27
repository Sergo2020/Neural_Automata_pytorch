import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

class CAModel(nn.Module):
    def __init__(self, channel_n, fire_rate, device, hidden_size=128):
        super(CAModel, self).__init__()

        self.device = device
        self.channel_n = channel_n
        self.hidden_size = hidden_size
        self.fire_rate = fire_rate
        self.to(self.device)

        self.fc0 = nn.Linear(self.channel_n * 3, self.hidden_size)
        self.fc1 = nn.Linear(self.hidden_size, self.channel_n, bias=False)
        with torch.no_grad():
            self.fc1.weight.zero_()

        self.max_pool = nn.MaxPool2d(3, 1, padding = 1)
        self.relu = nn.ReLU()
        self.sobel_x = None
        self.sobel_y = None

        self.prep_sobels()

    def prep_sobels(self):
        self.sobel_x = torch.from_numpy(np.outer([1, 2, 1], [-1, 0, 1]) / 8.0).float()
        self.sobel_y = self.sobel_x.T

        self.sobel_x = self.sobel_x.repeat(self.channel_n, 1, 1, 1).to(self.device)
        self.sobel_y = self.sobel_y.repeat(self.channel_n, 1, 1, 1).to(self.device)

    # Steps
    def alive(self, x):
        return self.max_pool(x[:, 3, :, :]) > 0.1

    def perceive(self, x, angle):

        c = np.cos(angle * np.pi / 180)
        s = np.sin(angle * np.pi / 180)
        w1 = c * self.sobel_x - s * self.sobel_y
        w2 = s * self.sobel_x + c * self.sobel_y

        y1 = F.conv2d(x, w1, padding=1, groups=self.channel_n)
        y2 = F.conv2d(x, w2, padding=1, groups=self.channel_n)

        y = torch.cat((x, y1, y2), 1)
        return y

    def update(self, x, fire_rate, angle):
        pre_life_mask = self.alive(x)

        delta = self.perceive(x, angle)
        delta = delta.transpose(1, 3)
        delta = self.fc0(delta)
        delta = self.relu(delta)
        delta = self.fc1(delta)
        delta = delta.transpose(1, 3)

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

    def forward(self, x, steps=1, fire_rate=None, angle=0.0):
        for step in range(steps):
            x = self.update(x, fire_rate, angle)
        return x
