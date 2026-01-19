import torch
import torch.nn as nn

from toolbox.tools import init_embeddings

import math
from types import SimpleNamespace
import numpy as np
import yaml


class LayerChunkedHypernetYaml(nn.Module):
    # Chunked Hypernetwork implementation
    # NOTE: Task embeddings are not internally maintained
    # NOTE: Chunk embeddings are internally maintained

    def __init__(
        self,
        config,
    ):
        super().__init__()

        self.config = config

        self.task_embedding_dim = config.task_embedding_dim
        self.chunk_embedding_dim = config.chunk_embedding_dim
        self.n_chunks = config.n_chunks
        self.hidden_layers = config.hnet_layers
        with open(f"{config.mnet_spec_path}/resnet{config.resnet_type}.yaml", "r") as f:
            self.mnet_spec = yaml.safe_load(f)
        self.in_dim = self.task_embedding_dim + self.chunk_embedding_dim

        # Chunk embedding generation
        self.chunk_embeddings = init_embeddings(
            self.n_chunks,
            self.chunk_embedding_dim,
            config.gpu,
        )

        # Mnet architecture info

        mnet_arch = SimpleNamespace()
        mnet_arch.n_layers = self.mnet_spec["n_layers"]
        mnet_arch.n_downsamples = self.mnet_spec["n_downsamples"]
        mnet_arch.weights = (
            self.mnet_spec["kernel_weights"] + self.mnet_spec["linear_weights"]
        )
        mnet_arch.n_weights = [np.array(w).prod() for w in mnet_arch.weights]
        mnet_arch.biases = (
            self.mnet_spec["kernel_biases"] + self.mnet_spec["linear_biases"]
        )
        mnet_arch.n_biases = mnet_arch.biases
        mnet_arch.net_n_params = sum(mnet_arch.n_weights) + sum(mnet_arch.n_biases)
        self.mnet_arch = mnet_arch

        self.calc_chunk_dim()  # Calculate chunk dimensions

        # Hypernetwork construction
        layers = []
        lr_list = [self.in_dim] + self.hidden_layers
        for i in range(len(lr_list) - 1):
            layers.append(nn.Linear(lr_list[i], lr_list[i + 1]))
            layers.append(nn.ReLU())

        self.hypernet_stack = nn.Sequential(*layers)

        self.weight_head = nn.Linear(
            self.hidden_layers[-1], mnet_arch.n_layers * self.weight_chunk_dim
        )
        self.bias_head = nn.Linear(
            self.hidden_layers[-1], mnet_arch.n_layers * self.bias_chunk_dim
        )
        self.norm_weight_head = nn.Linear(
            self.hidden_layers[-1], mnet_arch.n_layers * self.bias_chunk_dim
        )
        self.norm_bias_head = nn.Linear(
            self.hidden_layers[-1], mnet_arch.n_layers * self.bias_chunk_dim
        )
        self.skip_weight_head = nn.Linear(
            self.hidden_layers[-1], mnet_arch.n_downsamples * self.weight_chunk_dim
        )
        self.skip_bias_head = nn.Linear(
            self.hidden_layers[-1], mnet_arch.n_downsamples * self.bias_chunk_dim
        )

        # Hyperfan init for heads

        weight_head_w = self.weight_head.weight.data
        bias_head_w = self.bias_head.weight.data
        norm_weight_head_w = self.norm_weight_head.weight.data
        norm_bias_head_w = self.norm_bias_head.weight.data

        for layer in range(mnet_arch.n_layers - 1):
            # Fan in weight var calculation
            hnet_fan_in = self.hidden_layers[-1]
            kernel = self.mnet_spec["kernel_weights"][layer]
            mnet_fan_in = kernel[1] * kernel[2] * kernel[3]
            wgt_var = 1 / (hnet_fan_in * mnet_fan_in)
            init_val = torch.sqrt(3 * torch.tensor(wgt_var))
            w_initial = torch.distributions.uniform.Uniform(
                -init_val, init_val
            )  # Uniform Initialisation
            weight_head_w[
                layer * self.weight_chunk_dim : (layer + 1) * self.weight_chunk_dim, :
            ] = w_initial.sample((self.weight_chunk_dim, hnet_fan_in))
            # Fan in bias var calculation
            bias_fan_in = self.hidden_layers[-1]
            bias_var = 1 / bias_fan_in
            init_val = torch.sqrt(3 * torch.tensor(bias_var))
            b_initial = torch.distributions.uniform.Uniform(
                -init_val, init_val
            )  # Uniform Initialisation
            bias_head_w[
                layer * self.bias_chunk_dim : (layer + 1) * self.bias_chunk_dim
            ] = b_initial.sample((self.bias_chunk_dim, hnet_fan_in))
            norm_weight_head_w[
                layer * self.bias_chunk_dim : (layer + 1) * self.bias_chunk_dim
            ] = b_initial.sample((self.bias_chunk_dim, hnet_fan_in))
            norm_bias_head_w[
                layer * self.bias_chunk_dim : (layer + 1) * self.bias_chunk_dim
            ] = b_initial.sample((self.bias_chunk_dim, hnet_fan_in))

        # Last layer init
        # Fan in weight var calculation
        hnet_fan_in = self.hidden_layers[-1]
        mnet_fan_in = self.mnet_spec["linear_weights"][0][1]
        wgt_var = 1 / (hnet_fan_in * mnet_fan_in)
        init_val = torch.sqrt(3 * torch.tensor(wgt_var))
        w_initial = torch.distributions.uniform.Uniform(
            -init_val, init_val
        )  # Uniform Initialisation
        # w_initial = torch.distributions.normal.Normal(0, wgt_var) # Normal Initialisation
        weight_head_w[(self.mnet_arch.n_layers - 1) * self.weight_chunk_dim :] = (
            w_initial.sample((self.weight_chunk_dim, hnet_fan_in))
        )
        # Fan in bias var calculation
        bias_fan_in = self.hidden_layers[-1]
        bias_var = 1 / bias_fan_in
        init_val = torch.sqrt(3 * torch.tensor(bias_var))
        b_initial = torch.distributions.uniform.Uniform(
            -init_val, init_val
        )  # Uniform Initialisation
        bias_head_w[(self.mnet_arch.n_layers - 1) * self.bias_chunk_dim :] = (
            b_initial.sample((self.bias_chunk_dim, hnet_fan_in))
        )

    def calc_chunk_dim(self):
        max_mnet_weight_dim = max(self.mnet_arch.n_weights)
        max_mnet_bias_dim = max(self.mnet_arch.n_biases)
        self.weight_chunk_dim = math.ceil(max_mnet_weight_dim / self.n_chunks)
        self.bias_chunk_dim = math.ceil(max_mnet_bias_dim / self.n_chunks)

    def freeze_chunk_embeddings(self):
        for embedding in self.chunk_embeddings:
            embedding.requires_grad = False

    def get_arch(self):
        hypernet_output_size = 0
        for i in range(self.mnet_arch.n_layers):
            hypernet_output_size += self.weight_chunk_dim + 3 * self.bias_chunk_dim

        main_network_size = self.mnet_arch.net_n_params

        hypernet_size = 0
        for param in self.parameters():
            hypernet_size += param.numel()

        compression_ratio = hypernet_size / main_network_size

        arch_dict = {
            "task_embedding_dim": self.task_embedding_dim,
            "chunk_embedding_dim": self.chunk_embedding_dim,
            "n_chunks": self.n_chunks,
            "hidden_layers": self.hidden_layers,
            "weight_head": [
                self.weight_head.in_features,
                self.weight_head.out_features,
            ],
            "bias_head": [
                self.bias_head.in_features,
                self.bias_head.out_features,
            ],
            "norm_head": [
                self.norm_head.in_features,
                self.norm_head.out_features,
            ],
            "weight_chunk_dim": self.weight_chunk_dim,
            "bias_chunk_dim": self.bias_chunk_dim,
            "hypernet_output_size": hypernet_output_size,
            "hypernet_size": hypernet_size,
            "compression_ratio": compression_ratio,
        }
        return SimpleNamespace(**arch_dict)

    def forward(self, task_e):
        chunk_embedding_stack = torch.stack(self.chunk_embeddings, dim=0)
        task_embedding_stack = torch.stack([task_e] * self.n_chunks, dim=0)
        input_embedding_stack = torch.cat(
            (task_embedding_stack, chunk_embedding_stack), dim=1
        )
        representation_stack = self.hypernet_stack(input_embedding_stack)
        weights = self.weight_head(representation_stack)
        biases = self.bias_head(representation_stack)
        norm_w = self.norm_weight_head(representation_stack)
        norm_b = self.norm_bias_head(representation_stack)
        down_weights = self.skip_weight_head(representation_stack)
        down_biases = self.skip_bias_head(representation_stack)
        weights = weights.reshape(
            self.n_chunks,
            self.mnet_arch.n_layers,
            self.weight_chunk_dim,
        )
        biases = biases.reshape(
            self.n_chunks,
            self.mnet_arch.n_layers,
            self.bias_chunk_dim,
        )
        norm_w = norm_w.reshape(
            self.n_chunks,
            self.mnet_arch.n_layers,
            self.bias_chunk_dim,
        )
        norm_b = norm_b.reshape(
            self.n_chunks,
            self.mnet_arch.n_layers,
            self.bias_chunk_dim,
        )
        down_weights = down_weights.reshape(
            self.n_chunks,
            self.mnet_arch.n_downsamples,
            self.weight_chunk_dim,
        )
        down_biases = down_biases.reshape(
            self.n_chunks,
            self.mnet_arch.n_downsamples,
            self.bias_chunk_dim,
        )
        return weights, biases, norm_w, norm_b, down_weights, down_biases
