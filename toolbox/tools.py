import torch
from torch.utils.data import DataLoader

import json

# Function to create dataloader
def fetch_loaders(config, train_ds, test_ds):
    train_loaders = [
        DataLoader(
            ds,
            batch_size=config.batch_size,
            shuffle=True,
        )
        for ds in train_ds
    ]
    test_loaders = [
        DataLoader(
            ds,
            batch_size=config.batch_size,
            shuffle=True,
        )
        for ds in test_ds
    ]
    return train_loaders, test_loaders


# Function to init task and chunk embeddings
def init_embeddings(n_tasks, e_dim, gpu):
    embeddings = []
    for task in range(n_tasks):
        embedding = torch.randn(e_dim).to(find_device(gpu))
        # embedding = embedding / 100
        embedding.requires_grad = True
        embeddings.append(embedding)
    return embeddings


def fetch_task_embeddings(config):
    return init_embeddings(config.n_tasks, config.task_embedding_dim, config.gpu)


# Function that returns compute device
def find_device(gpu):
    return torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")


# Function to export architecture to yaml
def export_architecture(config, mnet_arch, hnet_arch):
    mnet_arch = mnet_arch.__dict__
    hnet_arch = hnet_arch.__dict__
    with open(f"logs/{config.expt_name}/mnet_arch.json", "w") as f:
        json.dump(mnet_arch, f)
    with open(f"logs/{config.expt_name}/hnet_arch.json", "w") as f:
        json.dump(hnet_arch, f)


def itr_merge(*itrs):
    for itr in itrs:
        for v in itr:
            yield v


