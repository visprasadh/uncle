from avalanche.benchmarks.classic import (
    PermutedMNIST,
    RotatedMNIST,
    SplitCIFAR100,
    SplitTinyImageNet,
)
from .five_datasets import get_dataset
import torchvision.transforms as transforms


def get_train_transform(config):
    t = [
        transforms.ToTensor(),
    ]
    if config.dataset.endswith("tinyimagenet"):
        t.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    elif config.dataset.endswith("cifar100"):
        t.extend(
            [
                transforms.Resize(64, antialias=True),
                transforms.Normalize(
                    (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
                ),
            ]
        )
    elif config.dataset.endswith("cifar10"):
        t.extend(
            [
                transforms.Resize(64, antialias=True),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
    elif config.dataset.endswith("fashionmnist") or config.dataset.endswith("notmnist"):
        t.extend(
            [
                transforms.Resize(64, antialias=True),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    elif config.dataset.endswith("mnist"):
        t.extend(
            [
                transforms.Resize(64, antialias=True),
                transforms.Normalize(
                    (0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)
                ),
            ]
        )
    elif config.dataset.endswith("svhn"):
        t.extend(
            [
                transforms.Resize(64, antialias=True),
                transforms.Normalize(
                    (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)
                ),
            ]
        )
    else:
        raise NotImplementedError
    return transforms.Compose(t)


def get_eval_transform(config):
    t = [
        transforms.ToTensor(),
    ]
    if config.dataset.endswith("tinyimagenet"):
        t.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    elif config.dataset.endswith("cifar100"):
        t.extend(
            [
                transforms.Resize(64, antialias=True),
                transforms.Normalize(
                    (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
                ),
            ]
        )
    elif config.dataset.endswith("cifar10"):
        t.extend(
            [
                transforms.Resize(64, antialias=True),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
    elif config.dataset.endswith("fashionmnist") or config.dataset.endswith("notmnist"):
        t.extend(
            [
                transforms.Resize(64, antialias=True),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    elif config.dataset.endswith("mnist"):
        t.extend(
            [
                transforms.Resize(64, antialias=True),
                transforms.Normalize(
                    (0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)
                ),
            ]
        )
    elif config.dataset.endswith("svhn"):
        t.extend(
            [
                transforms.Resize(64, antialias=True),
                transforms.Normalize(
                    (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)
                ),
            ]
        )
    else:
        raise NotImplementedError
    return transforms.Compose(t)


# Function to process Avalanche streams
def stream_processor(streams):
    train_stream = streams[0]
    test_stream = streams[1]
    train_datasets = []
    test_datasets = []

    for train_exp, test_exp in zip(train_stream, test_stream):
        tr_ds = train_exp.dataset
        te_ds = test_exp.dataset
        train_datasets.append(tr_ds)
        test_datasets.append(te_ds)

    return train_datasets, test_datasets


def get_dataset_info(dataset):
    return {
        "n_tasks": dataset.n_experiences,
        "n_classes": dataset.n_classes // dataset.n_experiences,
    }


# Permuted MNIST (Avalanche)
def fetch_permuted_mnist(config):
    dataset = PermutedMNIST(
        dataset_root=config.data_path,
        n_experiences=config.n_tasks,
        seed=int(config.seed),
        train_transform=transforms.Compose(
            [
                transforms.Lambda(lambda x: x.view(1, 28, 28)),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                transforms.Resize(64, antialias=True),
                transforms.Normalize(
                    (0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)
                ),
            ]
        ),
        eval_transform=transforms.Compose(
            [
                transforms.Lambda(lambda x: x.view(1, 28, 28)),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                transforms.Resize(64, antialias=True),
                transforms.Normalize(
                    (0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)
                ),
            ]
        ),
    )
    streams = dataset.train_stream, dataset.test_stream
    return stream_processor(streams), get_dataset_info(dataset)


# Rotated MNIST (Avalanche)
def fetch_rotated_mnist(config):
    rotations_list = list(range(0, 360, 360 // config.n_tasks))
    dataset = RotatedMNIST(
        dataset_root=config.data_path,
        n_experiences=config.n_tasks,
        rotations_list=rotations_list,
        seed=config.seed,
        train_transform=transforms.Compose(
            [
                transforms.Lambda(lambda x: x.view(1, 28, 28)),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                transforms.Resize(64, antialias=True),
                transforms.Normalize(
                    (0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)
                ),
            ]
        ),
        eval_transform=transforms.Compose(
            [
                transforms.Lambda(lambda x: x.view(1, 28, 28)),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                transforms.Resize(64, antialias=True),
                transforms.Normalize(
                    (0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)
                ),
            ]
        ),
    )
    streams = dataset.train_stream, dataset.test_stream
    return stream_processor(streams), get_dataset_info(dataset)


# Split CIFAR-100 (Avalanche)
def fetch_split_cifar100(config):
    dataset = SplitCIFAR100(
        dataset_root=config.data_path,
        n_experiences=config.n_tasks,
        seed=config.seed,
        class_ids_from_zero_in_each_exp=True,
        train_transform=get_train_transform(config),
        eval_transform=get_eval_transform(config),
    )
    streams = dataset.train_stream, dataset.test_stream
    return stream_processor(streams), get_dataset_info(dataset)


def fetch_split_tinyimagenet(config):
    dataset = SplitTinyImageNet(
        dataset_root=config.data_path,
        n_experiences=config.n_tasks,
        seed=config.seed,
        class_ids_from_zero_in_each_exp=True,
        train_transform=get_train_transform(config),
        eval_transform=get_eval_transform(config),
    )
    streams = dataset.train_stream, dataset.test_stream
    return stream_processor(streams), get_dataset_info(dataset)


def fetch_five_datasets(config):
    dataset_list = ["SVHN", "MNIST", "KMNIST", "NotMNIST", "FashionMNIST"]

    train_ds, test_ds = list(), list()

    for i in range(len(dataset_list)):
        config.dataset = dataset_list[i].lower()
        transform_train = get_train_transform(config)
        transform_val = get_eval_transform(config)
        dataset_train, dataset_test = get_dataset(
            dataset_list[i],
            transform_train,
            transform_val,
            config,
        )
        train_ds.append(dataset_train)
        test_ds.append(dataset_test)

    config.dataset = "five_datasets"
    return (train_ds, test_ds), None


def fetch_dataset(config):
    dataset = config.dataset
    if dataset == "permuted_mnist":
        return fetch_permuted_mnist(config)
    elif dataset == "rotated_mnist":
        return fetch_rotated_mnist(config)
    elif dataset == "split_cifar100":
        return fetch_split_cifar100(config)
    elif dataset == "split_tinyimagenet":
        return fetch_split_tinyimagenet(config)
    elif dataset == "five_datasets":
        return fetch_five_datasets(config)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")