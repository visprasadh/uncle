from toolbox.tools import find_device
from hypernet.hnet.chunked_hypernet import ChunkedHypernet
from hypernet.hnet.layer_chunked_hypernet import LayerChunkedHypernet
from hypernet.hnet.layer_chunked_hypernet_yaml import LayerChunkedHypernetYaml
from hypernet.mnet.cnn import CNN
from hypernet.mnet.resnet import Resnet


def fetch_mnet(config):
    device = find_device(config.gpu)
    if config.mnet_arch == "cnn":
        return CNN(config).to(device)

    elif config.mnet_arch == "resnet":
        return Resnet(config).to(device)

    else:
        raise ValueError("Invalid mnet architecture")


def fetch_hnet(config):
    device = find_device(config.gpu)

    mnet_arch = None

    if config.hnet_arch == "vanilla":
        raise NotImplementedError("Vanilla hypernet not implemented")

    elif config.hnet_arch == "chunked":
        raise NotImplementedError("Chunked hypernet not implemented")
        return ChunkedHypernet(config, mnet_arch).to(device)

    elif config.hnet_arch == "layer_chunked":
        return LayerChunkedHypernet(config, mnet_arch).to(device)

    elif config.hnet_arch == "layer_chunked_yaml":
        return LayerChunkedHypernetYaml(config).to(device)

    else:
        raise ValueError("Invalid hnet architecture")
