import torch

from model.VAE_DRO import train
from model.single_VAE import train_single_vae
from model.disentangle_models import train_DVAE
from utils.utils import show_VAE_by_TSNE

from model.DRO_Classifier import train_SC

import pickle

if __name__ == '__main__':
    latent_space, labels, models, cluster_pack = train_single_vae(12, './processed_data')
    for idx, m in enumerate(models):
        torch.save(m, f'./ckpts/vae_{idx}.pth')

    with open(f'./cluster.pkl', 'wb') as f:
        pickle.dump(cluster_pack, f)

    with open('./cluster.pkl', 'rb') as f:
        cluster_pack = pickle.load(f)
    train_SC(10000,cluster_pack)

    # torch.save(model,'./ckpts/vae.pth')
    # show_VAE_by_TSNE(latent_space, labels)
    # model= train_DVAE(20, './processed_data')
