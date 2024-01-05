from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from model.layers import *
from utils.data import get_loaders_per_class, get_dataloader, get_loaders_by_cluster_pack
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.utils import cluster_single_class


class StableClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super(StableClassifier, self).__init__()
        self.fea_extractor = nn.Sequential(
            Conv2dBlock(10, 4, 3, 1, 1),
            ResBlock(
                nn.Sequential(
                    Conv2dBlock(4, 4, 3, 1, 1, activation="relu"),
                    Conv2dBlock(4, 4, 3, 1, 1),
                ),
                version=1,
                # norm=nn.BatchNorm2d(4),
                # activation="relu",
            ),
            nn.AvgPool2d(2, 2),
            Conv2dBlock(4, 2, 3, 1, 1),
            ResBlock(
                nn.Sequential(
                    Conv2dBlock(2, 2, 3, 1, 1, norm="bn", activation="relu"),
                    Conv2dBlock(2, 2, 3, 1, 1, norm="bn"),
                ),
                version=1,
                # norm=nn.BatchNorm2d(4),
                # activation="relu",
            ),
            nn.AvgPool2d(2, 2),
            Conv2dBlock(2, 1, 3, 1, 1),
            nn.Flatten()
        )
        self.clf_head = nn.Sequential(
            LinearBlock(49, num_classes),
            nn.Softmax()
        )

        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x):
        fea = self.fea_extractor(x)
        x = self.clf_head(fea)
        return fea, x

    def loss(self, pred, y):
        return self.loss_func(pred, y)


def train_clf_loop(loaders, model, optimizer, device):
    batches = []
    for l in loaders:
        batch = l.next_batch()
        batches.append(batch)

    worst_acc = 0
    worst_loss = 0
    optimizer.zero_grad()
    for batch in batches:
        data = batch[0].to(device)
        tar = batch[1].to(device)
        _, pred = model(data)
        loss = model.loss(pred, tar)
        pred_tar = torch.argmax(pred, dim=1)
        acc = torch.sum(pred_tar == tar) / pred_tar.shape[0]
        if loss > worst_loss:
            worst_loss = loss
            worst_acc = acc
    worst_loss.backward()
    optimizer.step()

    return worst_loss.item(), worst_acc.item()


def validate_on_trainset_per_class(loader, model, device='cuda:0'):
    model.train()
    train_loss = 0
    batch_cnt = 0
    latent_list = []
    label_list = []
    channels = []
    filepaths = []
    with torch.no_grad():
        for data, tar, fp in tqdm(loader, total=len(loader)):
            data = data.to(device)
            tar = tar.to(device)

            z, pred = model(data)
            loss = model.loss(pred, tar)

            latent_list += list(z.detach().cpu().numpy())
            label_list += list(tar.detach().cpu().numpy())

            numpy_data = data.detach().cpu().numpy()
            for j in range(numpy_data.shape[0]):
                s = [np.sum(numpy_data[j, i, ::]) for i in range(10)]
                idx = np.argmax(np.asarray(s).astype(float))
                channels.append(idx)
            train_loss += loss.item()
            batch_cnt += 1
            filepaths += fp
            # channels += [loss.item() for _ in range(data.shape[0])]
    channels = np.asarray(channels)
    channels = np.round((channels - np.min(channels)) / (np.max(channels) - np.min(channels)) * 20)

    return latent_list, label_list, train_loss / batch_cnt, channels, filepaths


def validate_on_whole_trainset(loaders, model, optimizer, device='cuda:0'):
    cluster_res_dict = defaultdict(list)
    for i in range(10):
        latent_list, label_list, avg_train_loss, channels, file_paths = validate_on_trainset_per_class(
            loaders[i], model, device
        )
        logger.info(f'finish validate on class [{i}] with loss {avg_train_loss}')
        cluster_res = cluster_single_class(np.asarray(latent_list), np.asarray(channels))
        for cluster_group in cluster_res:
            fps = [file_paths[_] for _ in cluster_group]
            file_paths.append(fps)
            cluster_res_dict[f'{i}'].append(fps)
    return cluster_res_dict


def validate_loop(loader, model, device):
    batch_cnt = 0
    total_acc = 0
    model.eval()
    with torch.no_grad():
        for data, tar in tqdm(loader, total=len(loader)):
            data = data.to(device)
            tar = tar.to(device)
            _, pred = model(data)
            pred_tar = torch.argmax(pred, dim=1)
            acc = torch.sum(pred_tar == tar) / pred_tar.shape[0]
            total_acc += acc
            batch_cnt += 1
    model.train()

    return total_acc / batch_cnt


def train_SC(iteration: int,
             cluster_pack: str,
             data_root='./processed_data',
             device: str = 'cuda:0'):
    per_classes_loaders = get_loaders_per_class(data_root)
    loaders = get_loaders_by_cluster_pack(cluster_pack)
    logger.info(f'finish construct [{len(loaders)}] loaders')
    val_loader = get_dataloader(data_root, mode='val')
    logger.info(f'finish construct validation loaders')

    model = StableClassifier(10)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=1e-3)

    check_iter = 2
    check_iter2 = 2
    for iter_ in range(iteration):
        train_loss, acc = train_clf_loop(loaders, model, optimizer, device)
        logger.info(f'finish  {iter_} with [loss: {train_loss}|acc: {acc}]')
        if (iter_ % check_iter) == check_iter - 1:
            acc = validate_loop(val_loader, model, device)
            info = f'validate result at {iter_} with [acc: {acc}]'
            logger.info(f'\033[34m{info}\033[0m')
        if (iter_ % check_iter2) == check_iter2 - 1:
            cluster_pack = validate_on_whole_trainset(per_classes_loaders, model, device)

    return model
