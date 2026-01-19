import torch
import torch.nn as nn
import torch.nn.functional as F

from types import SimpleNamespace

import yaml

import numpy as np

from timm.layers import SelectAdaptivePool2d

from toolbox.tools import find_device

from toolbox.batchnorm import BatchNorm2d


class Resnet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_dim = config.mnet_in_dim  # of the format (C, H, W)
        self.type = config.resnet_type
        with open(f"{config.mnet_spec_path}/resnet{self.type}.yaml", "r") as f:
            self.spec = yaml.safe_load(f)
        self.batch_norm = nn.ModuleList()
        for layer in self.spec["layers"]:
            for block in self.spec["layers"][layer]:
                for item in self.spec["layers"][layer][block]:
                    if item["type"] == "BatchNorm2d":
                        bn = BatchNorm2d(item["num_features"])
                        bn.set_requires_grad(True)
                        bn.device = find_device(config.gpu)
                        self.batch_norm.append(bn)
        self.global_pool = SelectAdaptivePool2d(pool_type="avg")
        self.flatten = nn.Flatten()

    def forward(self, x, w, b, nw, nb, dw, db):
        w = w.reshape(self.spec["n_layers"], -1)
        b = b.reshape(self.spec["n_layers"], -1)
        nw = nw.reshape(self.spec["n_layers"], -1)
        nb = nb.reshape(self.spec["n_layers"], -1)
        dw = dw.reshape(self.spec["n_downsamples"], -1)
        db = db.reshape(self.spec["n_downsamples"], -1)
        spec = self.spec
        x_fork = 0
        ptr = 0
        norm_ptr = 0
        dw_ptr = 0
        for layer in spec["layers"]:
            for block in spec["layers"][layer]:
                for item in spec["layers"][layer][block]:
                    if item["type"] == "Conv2d":
                        kernel = spec["kernel_weights"][ptr]
                        bias = spec["kernel_biases"][ptr]
                        card = np.array(kernel).prod()
                        x = F.conv2d(
                            x,
                            w[ptr][:card].reshape(*kernel),
                            b[ptr][:bias],
                            stride=item["stride"],
                            padding=item["padding"],
                        )
                        ptr += 1
                    elif item["type"] == "BatchNorm2d":
                        card = spec["kernel_biases"][norm_ptr]
                        x = self.batch_norm[norm_ptr](
                            x, nw[norm_ptr][:card], nb[norm_ptr][:card]
                        )
                        norm_ptr += 1
                    elif item["type"] == "ReLU":
                        x = F.relu(x)
                    elif item["type"] == "MaxPool2d":
                        x = F.max_pool2d(
                            x,
                            item["kernel_size"],
                            stride=item["stride"],
                            padding=item["padding"],
                        )
                    elif item["type"] == "AdaptiveAvgPool2d":
                        x = self.global_pool(x)
                    elif item["type"] == "Linear":
                        wgt = spec["linear_weights"][0]
                        bias = spec["linear_biases"][0]
                        card = np.array(wgt).prod()
                        x = self.flatten(x)
                        x = F.linear(
                            x,
                            w[ptr][:card].reshape(*wgt),
                            b[ptr][:bias],
                        )
                    elif item["type"] == "Fork":
                        x_fork = x
                    elif item["type"] == "Add":
                        x = x + x_fork
                    elif item["type"] == "DownsampleAdd":
                        kernel = spec["downsample_weights"][dw_ptr]
                        # bias = spec["downsample_biases"][dw_ptr]
                        card = np.array(kernel).prod()
                        x_fork = F.conv2d(
                            x_fork,
                            dw[dw_ptr][:card].reshape(*kernel),
                            b[ptr][:bias],
                            stride=item["stride"],
                            padding=0,
                        )
                        x = x + x_fork
                        dw_ptr += 1
                    else:
                        raise ValueError(f"Unknown layer type {item['type']}")
        return x
