import random

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

import cnn_model as model


class Trainer(nn.Module):
    def __init__(self, hyperparams):
        super(Trainer, self).__init__()
        self.device = hyperparams['Device']
        self.batch_sz = hyperparams['Batch Size']
        self.h_dim = hyperparams['Hidden dim.']
        self.chan_n = hyperparams['Channels']
        self.precieve = hyperparams['Perceive']
        self.fire_rate = hyperparams['Fire rate']
        self.min_steps = hyperparams['Min. Steps']
        self.max_steps = hyperparams['Max. Steps']

        self.lr = hyperparams['Learning rate']
        self.lr_gamma = hyperparams['Learning gamma']

        self.CA = model.CAModel(self.chan_n, self.fire_rate, self.precieve, self.device, self.h_dim).to(self.device)
        self.mse_loss = model.MSE_rgb_loss().to(self.device)

        # Statistics
        self.train_loss = []

        # Optimizers
        self.optimizer = optim.AdamW(self.CA.parameters(), betas=(0.5, 0.5), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, self.lr_gamma)

    def train_model(self, pool_set, replace=True, update_pool=True):
        t_loss = torch.zeros(1).to(self.device)

        for idx, seeds, targets in pool_set.loader:
            self.optimizer.zero_grad()

            seeds = seeds.to(self.device)
            targets = targets.to(self.device)

            if replace:
                with torch.no_grad():
                    loss_s = self.mse_loss(seeds, targets).detach()
                    idx_loss = torch.argsort(loss_s, dim=0)
                    seeds = seeds[idx_loss]
                    seeds[-1] = pool_set.make_seed()

            steps = random.randint(self.min_steps, self.max_steps)
            out = self.CA(seeds, steps)
            loss = self.mse_loss(out, targets).mean()

            if update_pool:
                pool_set.pool[idx] = out.detach().cpu()

            loss.backward()
            nn.utils.clip_grad_norm_(self.CA.parameters(), max_norm=1.0, norm_type=2)
            self.optimizer.step()

            t_loss += loss

        self.train_loss.append(t_loss.cpu().item() / len(pool_set.loader))
        self.scheduler.step()
        return out.detach().cpu()

    def simulate_cells(self, seed, steps):
        res_list = []
        with torch.no_grad():
            seed = seed.to(self.device)
            out = seed.clone()
            for s in range(steps):
                out = self.CA.propogate(out)  # , None, angle)
                res_list.append(out.squeeze(0).cpu())

        return res_list

    def save_method(self, path, prefix):
        torch.save(self.CA.state_dict(), path / f'{prefix}_model.pt')
        torch.save(self.optimizer.state_dict(), path / f'{prefix}_opti.pt')
        torch.save(self.scheduler.state_dict(), path / f'{prefix}_sched.pt')

    def load_method(self, path, prefix):
        self.CA.load_state_dict(torch.load(path / f'{prefix}_model.pt'), strict=True)
        self.optimizer.load_state_dict(torch.load(path / f'{prefix}_opti.pt'))
        self.scheduler.load_state_dict(torch.load(path / f'{prefix}_sched.pt'))

    def plot_loss(self):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.train_loss) + 1), self.train_loss, c='r')
        plt.xlabel('Epochs')
        plt.ylabel('log loss')
        plt.yscale('log')
        plt.grid()
        plt.show()
