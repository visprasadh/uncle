import os

import os.path
import numpy as np
from typing import Any, Tuple

import torch
from torchvision import datasets
import codecs
import hashlib
import errno
from tqdm.auto import tqdm
from PIL import Image


def gen_bar_updater():
    pbar = tqdm(total=None)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def calculate_md5(fpath, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath, md5, **kwargs):
    return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath, md5=None):
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)


def makedir_exist_ok(dirpath):
    """
    Python2 support for os.makedirs(.., exist_ok=True)
    """
    try:
        os.makedirs(dirpath)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise


def download_url(url, root, filename=None, md5=None):
    """Download a file from a url and place it in root.
    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    from six.moves import urllib  # type: ignore

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    makedir_exist_ok(root)

    # downloads file
    if check_integrity(fpath, md5):
        print("Using downloaded and verified file: " + fpath)
    else:
        try:
            print("Downloading " + url + " to " + fpath)
            urllib.request.urlretrieve(url, fpath, reporthook=gen_bar_updater())
        except (urllib.error.URLError, IOError) as e:
            if url[:5] == "https":
                url = url.replace("https:", "http:")
                print(
                    "Failed download. Trying https -> http instead."
                    " Downloading " + url + " to " + fpath
                )
                urllib.request.urlretrieve(url, fpath, reporthook=gen_bar_updater())
            else:
                raise e


def get_int(b):
    return int(codecs.encode(b, "hex"), 16)


def open_maybe_compressed_file(path):
    """Return a file object that possibly decompresses 'path' on the fly.
    Decompression occurs when argument `path` is a string and ends with '.gz' or '.xz'.
    """
    if not isinstance(path, str):
        return path
    if path.endswith(".gz"):
        import gzip

        return gzip.open(path, "rb")
    if path.endswith(".xz"):
        import lzma

        return lzma.open(path, "rb")
    return open(path, "rb")


def read_sn3_pascalvincent_tensor(path, strict=True):
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
    Argument may be a filename, compressed filename, or file object.
    """
    # typemap
    if not hasattr(read_sn3_pascalvincent_tensor, "typemap"):
        read_sn3_pascalvincent_tensor.typemap = {
            8: (torch.uint8, np.uint8, np.uint8),
            9: (torch.int8, np.int8, np.int8),
            11: (torch.int16, np.dtype(">i2"), "i2"),
            12: (torch.int32, np.dtype(">i4"), "i4"),
            13: (torch.float32, np.dtype(">f4"), "f4"),
            14: (torch.float64, np.dtype(">f8"), "f8"),
        }
    # read
    with open_maybe_compressed_file(path) as f:
        data = f.read()
    # parse
    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256
    assert nd >= 1 and nd <= 3
    assert ty >= 8 and ty <= 14
    m = read_sn3_pascalvincent_tensor.typemap[ty]
    s = [get_int(data[4 * (i + 1) : 4 * (i + 2)]) for i in range(nd)]
    parsed = np.frombuffer(data, dtype=m[1], offset=(4 * (nd + 1)))
    assert parsed.shape[0] == np.prod(s) or not strict
    return torch.from_numpy(parsed.astype(m[2], copy=True)).view(*s)


def read_label_file(path):
    with open(path, "rb") as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert x.dtype == torch.uint8
    assert x.ndimension() == 1
    return x.long()


def read_image_file(path):
    with open(path, "rb") as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert x.dtype == torch.uint8
    assert x.ndimension() == 3
    return x


class MNIST_RGB(datasets.MNIST):
    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=False
    ):
        super(MNIST_RGB, self).__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self.train = train  # training set or test set

        if self._check_legacy_exist():
            self.data, self.targets = self._load_legacy_data()
            return

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        self.data, self.targets = self._load_data()

    def _check_legacy_exist(self):
        processed_folder_exists = os.path.exists(self.processed_folder)
        if not processed_folder_exists:
            return False

        return all(
            check_integrity(os.path.join(self.processed_folder, file))
            for file in (self.training_file, self.test_file)
        )

    def _load_legacy_data(self):
        # This is for BC only. We no longer cache the data in a custom binary, but simply read from the raw data
        # directly.
        data_file = self.training_file if self.train else self.test_file
        return torch.load(os.path.join(self.processed_folder, data_file))

    def _load_data(self):
        image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte"
        data = read_image_file(os.path.join(self.raw_folder, image_file))

        label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte"
        targets = read_label_file(os.path.join(self.raw_folder, label_file))

        return data, targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        try:
            img = Image.fromarray(img.numpy(), mode="L").convert("RGB")
        except Exception as e:
            print(e)
            pass

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, 1


class KMNIST(MNIST_RGB):
    """`Kuzushiji-MNIST <https://github.com/rois-codh/kmnist>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``KMNIST/processed/training.pt``
            and  ``KMNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    resources = [
        (
            "http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz",
            "6da9a790214d8f8ab97e552e6577677a",
        ),
        (
            "http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz",
            "949e75301f8813e0dc5800dcc320fc52",
        ),
        (
            "http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz",
            "3cf9ba584ccda5cd47140c280ad886eb",
        ),
        (
            "http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz",
            "9f55f451c0988b028eccdf7d5cc265ab",
        ),
    ]
    classes = ["o", "ki", "su", "tsu", "na", "ha", "ma", "ya", "re", "wo"]


class FashionMNIST(MNIST_RGB):
    """`Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``FashionMNIST/raw/train-images-idx3-ubyte``
            and  ``FashionMNIST/raw/t10k-images-idx3-ubyte`` exist.
        train (bool, optional): If True, creates dataset from ``train-images-idx3-ubyte``,
            otherwise from ``t10k-images-idx3-ubyte``.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    mirrors = ["http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"]

    resources = [
        ("train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
        ("train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe"),
        ("t10k-images-idx3-ubyte.gz", "bef4ecab320f06d8554ea6380940ec79"),
        ("t10k-labels-idx1-ubyte.gz", "bb300cfdad3c16e7a12a480ee83cd310"),
    ]
    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]


class NotMNIST(MNIST_RGB):
    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=False
    ):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        self.url = "https://github.com/facebookresearch/Adversarial-Continual-Learning/raw/main/data/notMNIST.zip"
        self.filename = "notMNIST.zip"

        fpath = os.path.join(root, self.filename)
        if not os.path.isfile(fpath):
            if not download:
                raise RuntimeError(
                    "Dataset not found. You can use download=True to download it"
                )
            else:
                print("Downloading from " + self.url)
                download_url(self.url, root, filename=self.filename)

        import zipfile

        zip_ref = zipfile.ZipFile(fpath, "r")
        zip_ref.extractall(root)
        zip_ref.close()

        if self.train:
            fpath = os.path.join(root, "notMNIST", "Train")

        else:
            fpath = os.path.join(root, "notMNIST", "Test")

        X, Y = [], []
        folders = os.listdir(fpath)

        for folder in folders:
            folder_path = os.path.join(fpath, folder)
            for ims in os.listdir(folder_path):
                try:
                    img_path = os.path.join(folder_path, ims)
                    X.append(np.array(Image.open(img_path).convert("RGB")))
                    Y.append(ord(folder) - 65)  # Folders are A-J so labels will be 0-9
                except Exception as e:
                    print(e)
                    print("File {}/{} is broken".format(folder, ims))
        self.data = np.array(X)
        self.targets = Y

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        try:
            img = Image.fromarray(img)
        except Exception as e:
            print(e)
            pass

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, 1


class CIFAR10(datasets.CIFAR10):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        return x, y, 1


class SVHN(datasets.SVHN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        return x, y, 1


def get_dataset(
    dataset,
    transform_train,
    transform_val,
    args,
):
    if dataset == "CIFAR10":
        dataset_train = CIFAR10(
            args.data_path, train=True, download=True, transform=transform_train
        )
        dataset_val = CIFAR10(
            args.data_path, train=False, download=True, transform=transform_val
        )

    elif dataset == "MNIST":
        dataset_train = MNIST_RGB(
            args.data_path, train=True, download=True, transform=transform_train
        )
        dataset_val = MNIST_RGB(
            args.data_path, train=False, download=True, transform=transform_val
        )

    elif dataset == "FashionMNIST":
        dataset_train = FashionMNIST(
            args.data_path, train=True, download=True, transform=transform_train
        )
        dataset_val = FashionMNIST(
            args.data_path, train=False, download=True, transform=transform_val
        )

    elif dataset == "SVHN":
        dataset_train = SVHN(
            args.data_path, split="train", download=True, transform=transform_train
        )
        dataset_val = SVHN(
            args.data_path, split="test", download=True, transform=transform_val
        )

    elif dataset == "NotMNIST":
        dataset_train = NotMNIST(
            args.data_path, train=True, download=True, transform=transform_train
        )
        dataset_val = NotMNIST(
            args.data_path, train=False, download=True, transform=transform_val
        )
    elif dataset == "KMNIST":
        dataset_train = KMNIST(
            args.data_path, train=True, download=True, transform=transform_train
        )
        dataset_val = KMNIST(
            args.data_path, train=False, download=True, transform=transform_val
        )
    return dataset_train, dataset_val