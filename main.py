"""
Main script for running continual learning and unlearning experiments.

This script sets up and executes experiments for continual learning and unlearning
using hypernetworks. It includes the following main components:

1. Configuration setup
2. Logger initialization
3. Dataset and model preparation
4. Training and unlearning loops
5. Evaluation and metric calculation
6. Results logging and model saving

Key functions:
- train(): Trains the model on a given task
- forget(): Performs unlearning for a specified task
- test(): Evaluates the model on a given task

The script uses a sequence of learn and unlearn instructions to guide the
experiment flow. Results are logged using prints and saved to CSV files.

Usage:
    python main.py <experiment_name>

Where <experiment_name> corresponds to a YAML config file in the config directory.
"""

import torch
import torch.nn as nn

from torch.optim.lr_scheduler import StepLR

import os
import sys
import copy
import random
import numpy as np
import pandas as pd
from loguru import logger

import toolbox.models as models
import toolbox.sequencer as sequencer
import toolbox.tools as tools
import toolbox.regularisation as regularisation
from toolbox.dataset import fetch_dataset
from toolbox.buffer import DERReplay
from toolbox.metrics import UnlearningMetricCalculator

import yaml
import json
from types import SimpleNamespace

import warnings

import time
import datetime

warnings.filterwarnings("ignore")

# Config -----------------------------------

# Get expt_name from command line argument

expt_name = sys.argv[1]
expt_series_name = "main"

with open(f"config/{expt_series_name}/{expt_name}.yaml", "r") as f:
    config_dict = yaml.safe_load(f)

config = SimpleNamespace(**config_dict)
config.expt_name = expt_name

# Logger config ----------------------------

os.makedirs(f"logs/{expt_series_name}/{expt_name}", exist_ok=True)
logger_id = logger.add(f"logs/{expt_series_name}/{expt_name}/output.log", level="TRACE")
logger.info(f"Experiment: {expt_name}")

config_dict = config.__dict__
with open(f"logs/{expt_series_name}/{config.expt_name}/config.json", "w") as f:
    json.dump(config_dict, f)

# Device config ----------------------------

gpu = config.gpu
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Seed config ------------------------------

seed = config.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
logger.info(f"Seed: {seed}")

# Dataset and Dataloader config ------------

datasets, _ = fetch_dataset(config)
train_ds, test_ds = datasets
train_loaders, test_loaders = tools.fetch_loaders(config, train_ds, test_ds)

# Model config -----------------------------

task_embeddings = tools.fetch_task_embeddings(config)
mnet = models.fetch_mnet(config)
hnet = models.fetch_hnet(config)

results_df = pd.DataFrame(
    columns=[
        "act",
        "task",
        "accuracy",
        "uniform",
        "mia_corr",
        "mia_conf",
        "mia_ent",
        "mia_prob",
        "ul_time",
    ]
)

criterion = nn.CrossEntropyLoss()

# Metric config

train_buffer = DERReplay(500) # Used for metric calculation
test_buffer = DERReplay(500)

metric_calculator = UnlearningMetricCalculator(
    device=device,
    hnet=True,
)

# Hyperparameters --------------------------

beta = config.beta  # Continual Learning Regularisation hyperparameter
gamma = config.gamma  # Unlearning Regularisation hyperparameter
partial = config.partial  # Boolean to use partial unlearning

learning_rate = config.lr  # Learning rate for the CL Optimizer
unlearning_rate = config.ulr  # Learning rate for the Unlearning Optimizer

n_forget_epochs = config.n_forget_epochs  # Number of epochs for unlearning
n_sampling = config.n_sampling  # Number of samples for unlearning


# Sequence preparation ---------------------
instruction_str = config.instruction_str
logger.info(f"Instruction sequence: {instruction_str}")

sequence = sequencer.generate_sequence(
    instruction_str,
    train_loaders,
    test_loaders,
)

# Training loop ----------------------------


def train(
    task,
    loader,
    hnet,
    previous_hnet,
    mnet,
    criterion,
    optimizer,
    beta,
    regen=False,
):
    hnet.train()
    for i, (x, y, _) in enumerate(loader):
        if regen and i > 5:  # Regeneration cutoff point
            break
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        e = task_embeddings[task]
        w, b, nw, nb, dw, db = hnet(e)
        y_pred = mnet(x, w, b, nw, nb, dw, db)
        loss = criterion(y_pred, y)
        if occupancy_tracker.sum() == 0:
            reg = torch.tensor(0.0, requires_grad=True).to(tools.find_device(gpu))
        else:
            reg = regularisation.output_reg(
                hnet,
                previous_hnet,
                task_embeddings,
                config.n_tasks,
                occupancy_tracker,
                gpu,
            )
        total_loss = loss + beta * reg 
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(hnet.parameters(), 1)
        # Checking accuracy of the model
        _, predicted = torch.max(y_pred, 1)
        correct = (predicted == y).sum().item()
        total = y.size(0)
        accuracy = correct / total
        optimizer.step()
        train_buffer.add(x, y, y_pred.detach(), task)
        print(
            f"loss: {loss.item():.4f}, reg: {reg.item():.4f}, total_loss: {total_loss.item():.4f}, train_accuracy {task}: {accuracy:.4f}"
        )
        if i % 100 == 0:
            logger.debug(
                f"Loss: {loss.item():.4f} | Reg: {reg.item():.4f} | Total loss: {total_loss.item():.4f} | Accuracy: {accuracy:.4f}"
            )


def forget(
    forget_task,
    hnet,
    previous_hnet,
    optimizer,
    forget_epochs,
    n_sampling,
    gamma,
    partial,
):
    print(f"Forgetting task {forget_task} for {forget_epochs} epochs")
    hnet.train()
    for _ in range(forget_epochs):
        optimizer.zero_grad()
        rreg = regularisation.remember_reg(
            hnet,
            previous_hnet,
            task_embeddings,
            forget_task,
            config.n_tasks,
            occupancy_tracker,
            gpu,
        )
        freg = regularisation.noisy_forget(
            hnet, forget_task, task_embeddings, n_sampling, partial
        )
        loss = rreg + gamma * freg
        logger.debug(f"Remember reg: {rreg} | Forget reg: {freg.item()}")
        logger.debug(f"Loss: {loss.item()}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(hnet.parameters(), 1)
        optimizer.step()


def test(act, task, loader, hnet, mnet):
    hnet.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (x, y, _) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            e = task_embeddings[task]
            w, b, nw, nb, dw, db = hnet(e)
            y_pred = mnet(x, w, b, nw, nb, dw, db)
            _, predicted = torch.max(y_pred, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            test_buffer.add(x, y, y_pred, task)
    accuracy = correct / total
    logger.info(f"Accuracy: {accuracy * 100}")
    results_df.loc[len(results_df)] = [act, task, accuracy * 100] + [0.0] * 6
    return accuracy * 100

# Driver code --------------------------------------------------------------------------------------

occupancy_tracker = torch.zeros(config.n_tasks)
previous_hnet = None

acc_matrix = np.zeros((len(sequence), config.n_tasks))
current_forget_epochs = config.n_forget_epochs

for seq_idx, item in enumerate(sequence):
    # Learn
    if item["act"] == "learn":
        task, loader = item["train_package"]
        logger.info(f"Learning task {task}")
        if config.chunk and occupancy_tracker.sum() > 0:
            hnet.freeze_chunk_embeddings()
        optimizer = torch.optim.Adam(
            list(hnet.parameters()) + [task_embeddings[task]],
            lr=learning_rate,
        )
        lr_scheduler = StepLR(
            optimizer,
            step_size=25,
            gamma=0.5,
        )
        for epoch in range(config.n_epochs):
            logger.info(
                f"Epoch {epoch}, Learning rate {optimizer.param_groups[0]['lr']}"
            )
            train(
                task,
                loader,
                hnet,
                previous_hnet,
                mnet,
                criterion,
                optimizer,
                beta,
            )
            lr_scheduler.step()
        occupancy_tracker[task] = 1
        previous_hnet = copy.deepcopy(hnet)
        for param in previous_hnet.parameters():
            param.requires_grad = False

        for i, (t, test_loader) in enumerate(item["test_package"]):
            logger.info(f"Testing task {t}")
            acc = test(f"L{task}", t, test_loader, hnet, mnet)
            acc_matrix[seq_idx][i] = acc

    # Unlearn
    elif item["act"] == "unlearn":
        forget_task, forget_train_loader, forget_test_loader = item["forget_package"]
        logger.info(f"Forgetting task {forget_task}")
        optimizer = torch.optim.Adam(
            hnet.parameters(),
            lr=unlearning_rate,
        )
        start_time = time.time()
        forget(
            forget_task,
            hnet,
            previous_hnet,
            optimizer,
            current_forget_epochs,
            n_sampling,
            gamma,
            partial=partial,
        )
        end_time = time.time()
        unlearning_time = end_time - start_time
        logger.info(f"Unlearning time: {datetime.timedelta(seconds=unlearning_time)}")
        occupancy_tracker[forget_task] = 0
        previous_hnet = copy.deepcopy(hnet)
        for param in previous_hnet.parameters():
            param.requires_grad = False

        for i, (t, test_loader) in enumerate(item["test_package"]):
            logger.info(f"Testing task {t}")
            acc = test(f"U{forget_task}", t, test_loader, hnet, mnet)
            acc_matrix[seq_idx][i] = acc

        train_buffer.remove(forget_task)
        test_buffer.remove(forget_task)

        # Calculate Unlearning Score
        logger.info("Calculating Unlearning Score")
        ul_uniform, mia = metric_calculator.get_the_metrics(
            (task_embeddings, hnet, mnet),
            forget_task,
            forget_train_loader,
            forget_test_loader,
            train_buffer,
            test_buffer,
        )
        logger.info(f"UL Score Uniform: {ul_uniform}, MIA: {mia}")
        results_df.loc[len(results_df)] = [
            f"U{forget_task}",
            forget_task,
            0.0,
            ul_uniform.item(),
            mia["correctness"],
            mia["confidence"],
            mia["entropy"],
            mia["prob"],
            unlearning_time,
        ]

        # Reduce the number of forget epochs by 10% for the next unlearning operation
        current_forget_epochs = int(current_forget_epochs * 0.9)
        logger.info(f"Updated forget epochs for next unlearning: {current_forget_epochs}")

results_df.to_csv(f"logs/{expt_series_name}/{config.expt_name}/results.csv")
np.savetxt(
    f"logs/{expt_series_name}/{config.expt_name}/acc_matrix.csv",
    acc_matrix,
    delimiter=",",
    fmt="%.4f",
)

# Save the model
weights = {
    "hnet": hnet.state_dict(),
    "task_e": task_embeddings,
    "chunk_e": hnet.chunk_embeddings,
}

torch.save(weights, f"logs/{expt_series_name}/{config.expt_name}/weights.pth")

print("Experiment finished")
