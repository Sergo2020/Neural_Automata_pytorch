import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class CAModel(nn.Module):
    def __init__(self, channel_n, fire_rate, perceive, device, hidden_size=128):
        super(CAModel, self).__init__()

        self.device = device
        self.channel_n = channel_n
        self.hidden_size = hidden_size
        self.fire_rate = fire_rate
        self.perceive_flag = perceive
        self.alive_th = 0.1

        self.prep_sobels()

        self.conv1 = nn.Conv2d(self.channel_n * 3, self.hidden_size, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(self.hidden_size, self.channel_n, 1, 1)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def prep_sobels(self):
        self.sobel_x = torch.from_numpy(np.outer([1, 2, 1], [-1, 0, 1]) / 8.0).float()
        self.sobel_y = self.sobel_x.T

        self.sobel_x = self.sobel_x.repeat(self.channel_n, 1, 1, 1).to(self.device)
        self.sobel_y = self.sobel_y.repeat(self.channel_n, 1, 1, 1).to(self.device)

    def stochastic_update(self, d):
        d_size = d[:, 3:4, :, :].size()
        stoch_grid = torch.rand(d_size).to(self.device) > self.fire_rate

        return d * stoch_grid

    def alive_mask(self, x):
        alpha = x[:, 3:4, :, :]
        mask = self.max_pool(alpha) >= self.alive_th
        return mask

    def perceive(self, x, angle=0):

        c = np.cos(angle * np.pi / 180)
        s = np.sin(angle * np.pi / 180)
        w1 = c * self.sobel_x - s * self.sobel_y
        w2 = s * self.sobel_x + c * self.sobel_y

        y1 = F.conv2d(x, w1, padding=1, groups=self.channel_n)
        y2 = F.conv2d(x, w2, padding=1, groups=self.channel_n)

        y = torch.cat((x, y1, y2), 1)
        return y

    def propogate(self, x):

        alive_prev = self.alive_mask(x)

        if self.perceive_flag:
            delta = self.perceive(x)
        else:
            delta = x

        delta = self.relu(self.conv1(delta))
        delta = self.conv2(delta)
        delta = self.stochastic_update(delta)

        x = x + delta

        alive = self.alive_mask(x)

        x *= (alive_prev & alive).float()
        return x

    def forward(self, x, steps):
        for _ in range(steps):
            x = self.propogate(x)

        return x

class MSE_rgb_loss(nn.Module):
    def __init__(self):
        super(MSE_rgb_loss, self).__init__()

        self.mse = nn.MSELoss(reduction='none')

    def forward(self, out, target):
        return self.mse(out[:, :4], target[:, :4]).mean(dim=(-3, -2, -1))