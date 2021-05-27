import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class Single_image(Dataset):
    def __init__(self, hyper):
        self.img_path = hyper['Image path']
        self.channel_n = hyper['Channels']
        self.padding = hyper['Target padding']
        self.pool_size = hyper['Pool size']
        self.batch_size = hyper['Batch Size']

        self.normilizer = transforms.Compose([transforms.Pad(self.padding),
                                              transforms.ToTensor()])

        self.img_pil = self.load_img()
        self.target = self.prep_img(self.img_pil, 0)
        self.target_pad = self.prep_img(self.img_pil, self.padding)
        self.seed = self.make_seed()
        self.pool = self.init_pool()
        self.dummy_pool = self.pool.clone()
        self.loader = self.init_loader()

    def init_pool(self):
        pool = self.seed.repeat(self.pool_size, 1, 1, 1)
        return pool

    def make_seed(self):
        h, w = self.target_pad.size()[-2:]
        x = torch.zeros((1, self.channel_n, h, w))
        x[:, 3:, x.size()[2] // 2, x.size()[3] // 2] = 1.0
        return x

    def load_img(self):
        img_pil = Image.open(self.img_path)

        return img_pil

    @staticmethod
    def prep_img(img_pil, pad):
        if pad > 0:
            normilizer = transforms.Compose([transforms.Pad(pad),
                                             transforms.ToTensor()])
        else:
            normilizer = transforms.ToTensor()
        img_tens = normilizer(img_pil)
        img_tens[:3] *= img_tens[3:4]  # Multiply by alpha to preserve object shape
        return img_tens

    def update_pool(self):
        self.pool = self.dummy_pool.clone()
        self.dummy_pool = self.init_pool()

    def init_loader(self):
        return DataLoader(self, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def save_data(self, save_dir, prefix):
        torch.save(self, save_dir / f'{prefix}_data.pt')

    def __len__(self):
        return self.pool_size

    def __getitem__(self, idx):
        seed = self.pool[idx]

        return idx, seed, self.target_pad
