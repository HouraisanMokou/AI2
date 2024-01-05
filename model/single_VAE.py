import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from collections import defaultdict

from utils.utils import show_VAE_by_TSNE, show_VAE_by_multi_TSNE, show_VAE_by_single_TSNE, cluster_single_class

from utils.data import CostumedDataset, get_dataloader, get_loaders_per_class
from tqdm import tqdm
from loguru import logger


class VAE_Encoder(nn.Module):
    def __init__(self, chs, kernel_size, z_dim):
        super(VAE_Encoder, self).__init__()
        self.conv1 = nn.Conv2d(10, chs[0], kernel_size)
        self.conv2 = nn.Conv2d(chs[0], chs[1], kernel_size)
        self.conv3 = nn.Conv2d(chs[1], chs[2], kernel_size)
        self.linear1 = nn.Linear(8192, 256)
        self.mu_head = nn.Linear(256, z_dim)
        self.var_head = nn.Linear(256, z_dim)

    def forward(self, x):
        fea = F.relu(self.conv1(x))
        fea = F.relu(self.conv2(fea))
        fea = F.relu(self.conv3(fea))
        fea = fea.reshape((fea.shape[0], -1))
        fea = F.relu(self.linear1(fea))
        mu = self.mu_head(fea)
        var = self.var_head(fea)
        return mu, var


class VAE_Decoder(nn.Module):
    def __init__(self, chs, kernel_size, z_dim):
        super(VAE_Decoder, self).__init__()
        self.chs = chs
        self.linear1 = nn.Linear(z_dim, 256)
        self.linear2 = nn.Linear(256, 8192)
        self.conv3 = nn.ConvTranspose2d(chs[-1], chs[-2], kernel_size)
        self.conv2 = nn.ConvTranspose2d(chs[-2], chs[-3], kernel_size)
        self.conv1 = nn.ConvTranspose2d(chs[-3], 10, kernel_size)

    def forward(self, z):
        fea = F.relu(self.linear1(z))
        fea = F.relu(self.linear2(fea))
        fea = fea.reshape(-1, self.chs[-1], 16, 16)
        fea = F.relu(self.conv3(fea))
        fea = F.relu(self.conv2(fea))
        img = self.conv1(fea)
        return F.sigmoid(img)


class VAE(nn.Module):
    def __init__(self, chs, z_dim):
        super(VAE, self).__init__()

        # encoder part
        self.encoder_layer = VAE_Encoder(chs, 5, z_dim)
        # decoder part
        self.decoder_layer = VAE_Decoder(chs, 5, z_dim)

    def encoder(self, x):
        return self.encoder_layer(x)  # mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def decoder(self, z):
        return self.decoder_layer(z)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        img = self.decoder(z)
        return img, mu, log_var, z


def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x.reshape(-1, 28 * 28), x.reshape(-1, 28 * 28), reduction='sum') \
          / recon_x.shape[0]
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / recon_x.shape[0]
    return BCE + KLD


def test(model, loader, device='cuda:0'):
    latent_list = []
    label_list = []
    sample_list = []
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(loader):
            data = data.to(device)
            recon_batch, mu, log_var, z = model(data)
            sample_list += list(data.cpu().numpy())
            latent_list += list(z.cpu().numpy())
            label_list += list(_.cpu().numpy())
    return np.asarray(sample_list), np.asarray(latent_list), np.asarray(label_list)


def train_loop(loader, model, optimizer, device='cuda:0'):
    model.train()
    train_loss = 0
    batch_cnt = 0
    latent_list = []
    label_list = []
    channels = []
    filepaths = []
    for data, tar, fp in tqdm(loader, total=len(loader)):
        data = data.to(device)
        optimizer.zero_grad()

        recon_batch, mu, log_var, z = model(data)
        loss = loss_function(recon_batch, data, mu, log_var)

        latent_list += list(z.detach().cpu().numpy())
        label_list += list(tar.detach().cpu().numpy())
        loss.backward()
        numpy_data = data.detach().cpu().numpy()
        for j in range(numpy_data.shape[0]):
            s = [np.sum(numpy_data[j, i, ::]) for i in range(10)]
            idx = np.argmax(np.asarray(s).astype(float))
            channels.append(idx)
        train_loss += loss.item()
        batch_cnt += 1
        filepaths += fp
        optimizer.step()
    return latent_list, label_list, train_loss / batch_cnt, channels, filepaths


def train_single_vae(epoch, data_root, device='cuda:0'):
    loaders = get_loaders_per_class(data_root)

    models = []

    total_latent_list = []
    total_label_list = []
    cluster_res_dict = defaultdict(list)
    for i in range(10):
        model = VAE(chs=[32, 32, 32], z_dim=6)
        model.to(device)

        latent_list, label_list, channels, file_paths = None, None, None, None

        optimizer = Adam(model.parameters(), lr=1e-3)
        for e in range(epoch):
            latent_list, label_list, avg_train_loss, channels, file_paths = train_loop(loaders[i], model, optimizer,
                                                                                       device)
            logger.info(f'finish epoch {e} with loss {avg_train_loss}')

        cluster_res = cluster_single_class(np.asarray(latent_list), np.asarray(channels))
        for cluster_group in cluster_res:
            fps = [file_paths[_] for _ in cluster_group]
            file_paths.append(fps)
            cluster_res_dict[f'{i}'].append(fps)

        models.append(model)
        total_latent_list.append(np.array(latent_list))
        total_label_list.append(np.array(label_list))
        info = f'vae of class [{i}] training finish'
        logger.info(f'\033[34m{info}\033[0m')

    return total_latent_list, total_label_list, models, cluster_res_dict
