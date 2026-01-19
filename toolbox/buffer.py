# Description: Replay buffer for continual learning

import numpy as np
import random
import torch
from torch.utils.data import Dataset


class ReplayBuffer(Dataset):
    def __init__(self, samples_per_task):
        self.samples_per_task = samples_per_task
        self.num_seen_examples = 0
        self.seen_datasets = {}
        self.sample_tracker = []

    def __len__(self):
        return self.num_seen_examples

    def add_data(self, dataset, task_label=None):
        if len(dataset) < self.samples_per_task:
            self.seen_datasets.update({task_label: dataset})
            self.num_seen_examples += len(dataset)
            self.sample_tracker.extend([(task_label, i) for i in range(len(dataset))])
        else:
            # generate random indices
            indices = np.random.choice(
                len(dataset),
                self.samples_per_task,
                replace=False,
            )
            self.seen_datasets.update({task_label: dataset})
            self.num_seen_examples += self.samples_per_task
            self.sample_tracker.extend([(task_label, i) for i in indices])
        random.shuffle(self.sample_tracker)

    def delete_data(self, task_labels):
        for i in task_labels:
            del self.seen_datasets[i]
        tracker = []
        for i, j in self.sample_tracker:
            if i in task_labels:
                self.num_seen_examples -= 1
            else:
                tracker.append((i, j))
        self.sample_tracker = tracker
        random.shuffle(self.sample_tracker)

    def __getitem__(self, index):
        dataset_idx, sample_idx = self.sample_tracker[index]
        x, y, _ = self.seen_datasets[dataset_idx][sample_idx]
        task_label = dataset_idx
        return x, y, task_label


class DERReplay(Dataset):
    """
    A dataset wrapper used as a memory to store the data
    """

    def __init__(self, buffer_size):
        super(DERReplay, self).__init__()
        self.buffer_size = buffer_size
        self.buffer = {}
        self.items = []

    def __len__(self):
        if not self.buffer:
            return 0
        else:
            return len(self.items)

    def add(self, x, y, h, t):
        x = x.cpu()
        y = y.cpu()
        h = h.cpu()
        dim_h = h.shape[-1]
        x_shape = x.shape[1:]

        if t not in self.buffer:
            self.buffer[t] = {
                "X": torch.zeros([self.buffer_size] + list(x_shape)),
                "Y": torch.zeros(self.buffer_size).long(),
                "H": torch.zeros([self.buffer_size] + [dim_h]),
                "num_seen": 0,
            }

        n = x.shape[0]
        for i in range(n):
            self.buffer[t]["num_seen"] += 1

            if self.buffer[t]["num_seen"] <= self.buffer_size:
                idx = self.buffer[t]["num_seen"] - 1
            else:
                rand = np.random.randint(0, self.buffer[t]["num_seen"])
                idx = rand if rand < self.buffer_size else -1

            self.buffer[t]["X"][idx] = x[i]
            self.buffer[t]["Y"][idx] = y[i]
            self.buffer[t]["H"][idx] = h[i]

        self._make_dataset_requirements()

    def sample(self, n, exclude=[]):
        nb = self.__len__()
        if nb == 0:
            return None, None

        X, Y, H = [], [], []
        for t, v in self.buffer.items():
            if t in exclude:
                continue
            idx = torch.randperm(min(v["num_seen"], v["X"].shape[0]))[
                : min(min(n, v["num_seen"]), v["X"].shape[0])
            ]
            X.append(v["X"][idx])
            Y.append(v["Y"][idx])
            H.append(v["H"][idx])
        return (torch.cat(X, 0), torch.cat(Y, 0), torch.cat(H, 0))

    def sample_task(self, n, task_id):
        X, Y, H = [], [], []
        assert task_id in self.buffer, f"[ERROR] not found {task_id} in buffer"
        v = self.buffer[task_id]
        idx = torch.randperm(min(v["num_seen"], v["X"].shape[0]))[
            : min(min(n, v["num_seen"]), v["X"].shape[0])
        ]
        X.append(v["X"][idx])
        Y.append(v["Y"][idx])
        H.append(v["H"][idx])
        return (torch.cat(X, 0), torch.cat(Y, 0), torch.cat(H, 0))

    def remove(self, t):
        X = self.buffer[t]["X"]
        Y = self.buffer[t]["Y"]
        H = self.buffer[t]["H"]
        del self.buffer[t]
        self._make_dataset_requirements()
        return X, Y, H

    def _make_dataset_requirements(self):
        self.items = []
        for t, v in self.buffer.items():
            n_samples = min(v["num_seen"], v["X"].shape[0])
            self.items.extend([(t, i) for i in range(n_samples)])
        random.shuffle(self.items)

    def __getitem__(self, index):
        task, sample_idx = self.items[index]
        x = self.buffer[task]["X"][sample_idx]
        y = self.buffer[task]["Y"][sample_idx]
        return x.squeeze(), y.squeeze(), task