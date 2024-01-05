import os

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Sampler


class CostumedVAEDataset(Dataset):
    def __init__(self, filenames, targets=None, testing=False):
        self.targets = targets
        self.filenames = filenames
        self.testing = testing

        means = 0
        stds = 1
        self.transform = transforms.Compose([
            transforms.Normalize(means, stds)
        ])

        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop((28, 28), scale=(0.8, 1.0))
        ])

    def __getitem__(self, idx):
        fp = self.filenames[idx]
        image = np.load(fp)
        if self.testing == True:
            return self.transform(image)
        else:
            # return self.train_transform(self.transform(torch.tensor(image))), torch.tensor(self.targets[idx])
            return self.transform(torch.tensor(image)), torch.tensor(self.targets[idx], dtype=torch.int64), fp

    def __len__(self):
        return len(self.filenames)


class CostumedDataset(Dataset):
    def __init__(self, filenames, targets=None, testing=False):
        self.targets = targets
        self.filenames = filenames
        self.testing = testing

        means = 0
        stds = 1
        self.transform = transforms.Compose([
            transforms.Normalize(means, stds)
        ])

        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop((28, 28), scale=(0.8, 1.0))
        ])

    def __getitem__(self, idx):
        fp = self.filenames[idx]
        image = np.load(fp)
        if self.testing == True:
            return self.transform(image)
        else:
            # return self.train_transform(self.transform(torch.tensor(image))), torch.tensor(self.targets[idx])
            return self.transform(torch.tensor(image)), torch.tensor(self.targets[idx], dtype=torch.int64)

    def __len__(self):
        return len(self.filenames)


class InfiniteLoader:
    def __init__(self, loader):
        self.iterator = iter(loader)
        self.loader = loader

    def next_batch(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            batch = next(self.iterator)
        return batch


def get_meta(root, testing=False):
    print('getting meta')
    filenames = []
    if testing == False:
        targets = []
        for cn in os.listdir(root):
            cn_path = os.path.join(root, cn)
            cn_filenames = os.listdir(cn_path)
            targets += list(np.ones(len(cn_filenames)) * int(cn))
            filenames += [os.path.join(cn_path, _) for _ in cn_filenames]

        targets = np.asarray(targets)
        return filenames, targets
    else:
        filenames = [os.path.join(root, _) for _ in os.listdir(root)]
        return filenames, None


def get_single_class_meta(root, classname):  # only for train or val
    print('getting meta')
    filenames = []
    targets = []

    cn_path = os.path.join(root, str(classname))
    cn_filenames = os.listdir(cn_path)
    targets += list(np.ones(len(cn_filenames)) * int(classname))
    filenames += [os.path.join(cn_path, _) for _ in cn_filenames]

    targets = np.asarray(targets)
    return filenames, targets


def get_loaders_per_class(data_root, batch_size=128, num_worker=0, mode='train', sampler=None):
    root = data_root + '/' + mode
    kwargs = {
        'batch_size': batch_size,
        'num_workers': num_worker,
        'pin_memory': True,
        'shuffle': mode != 'test'
    }
    loaders = []
    for i in range(10):
        filenames, targets = get_single_class_meta(root, i)
        dataset = CostumedVAEDataset(filenames, targets)
        kwargs['dataset'] = dataset
        if sampler is not None:
            kwargs['samplers'] = sampler
        dataloader = DataLoader(**kwargs)
        loaders.append(dataloader)
    return loaders


def get_dataloader(data_root, batch_size=128, num_worker=0, mode='train', sampler=None):
    root = data_root + '/' + mode
    filenames, targets = get_meta(root, testing=mode == 'test')
    dataset = CostumedDataset(filenames, targets)
    kwargs = {
        'dataset': dataset,
        'batch_size': batch_size,
        'num_workers': num_worker,
        'pin_memory': True,
        'shuffle': mode != 'test'
    }
    if sampler is not None:
        kwargs['samplers'] = sampler
    dataloader = DataLoader(**kwargs)
    return dataloader


def get_dual_loaders(data_root, batch_size=128, num_worker=4, mode='train', sampler=None):
    root = data_root + '/' + mode
    filenames, targets = get_meta(root, testing=mode == 'test')
    dataset = CostumedDataset(filenames, targets)
    kwargs = {
        'dataset': dataset,
        'batch_size': batch_size,
        'num_workers': num_worker,
        'pin_memory': True,
        'shuffle': mode != 'test'
    }
    if sampler is not None:
        kwargs['samplers'] = sampler
    dataloader1 = DataLoader(**kwargs)
    dataloader2 = DataLoader(**kwargs)
    return [dataloader1, dataloader2]


def get_loaders_by_cluster_pack(cluster_pack, batch_size=128, num_worker=0, mode='train', sampler=None):
    '''
    :param cluster_pack: {
        '0': [[filenames...],[filenames...],...],
        ...
    }
    '''
    kwargs = {
        'dataset': None,
        'batch_size': batch_size,
        'num_workers': num_worker,
        'pin_memory': True,
        'shuffle': mode != 'test'
    }
    loaders = []
    for k in cluster_pack.keys():
        clusters = cluster_pack[k]
        for fns in clusters:
            targets = np.ones(len(fns)) * int(k)
            dataset = CostumedDataset(fns, targets)
            kwargs['dataset'] = dataset
            loader = InfiniteLoader(DataLoader(**kwargs))
            loaders.append(loader)
    return loaders
