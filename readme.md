# UnCLe: An Unlearning framework for Continual Learning

## Project Description

Code repostory for the ICLR submisssion UnCLe

## How to Run the Project

1. Install dependencies (refer to the requirements.txt file)
2. Prepare your dataset(s) in the `./data` directory
3. Configure your experiment using a YAML file (see Configuration section)
4. Run the run.sh script

# Configuration

This section outlines the naming convention for configuration files in our UnCLe (Unlearning framework for Continual Learning) project.

The subfolder main contains all the configuration files required for the main experiments

## File Name Format

Configuration files should follow this naming pattern:

`<dataset>_<sequence>_<seed>.yaml`

Where:
- `<dataset>`: Specifies the dataset used for the experiment (e.g., 5d, c100, pm, tiny)
    - `5d`: Represents the "Five Datasets" experiment, which includes a combination of five different datasets
    - `c100`: Refers to the CIFAR-100 dataset
    - `pm`: Stands for Permuted MNIST, a variant of the MNIST dataset where pixels are permuted
    - `tiny`: Indicates the TinyImageNet dataset, a subset of ImageNet with 200 classes

- `<sequence>`: Indicates the sequence of tasks and their order (e.g., 5a, 10b, 20c)

- The number in the sequence corresponds to the number of tasks:
    - 5 series (e.g., 5a, 5b, 5c): Used for the Five Datasets (5d) experiment
    - 10 series (e.g., 10a, 10b, 10c): Used for Permuted MNIST (pm) and CIFAR-100 (c100) experiments
    - 20 series (e.g., 20a, 20b, 20c): Used for TinyImageNet (tiny) experiments

- The letter (a, b, c) denotes different task orderings or variations within the same number of tasks

- `<seed>`: Represents the random seed used for reproducibility (e.g., s5, s17, s42)

## Examples

Here are some example config file names:

- `5d_5a_s5.yaml`: Five datasets, sequence 5a, seed 5
- `c100_10b_s17.yaml`: CIFAR100 dataset, sequence 10b, seed 17
- `pm_10a_s42.yaml`: Permuted MNIST dataset, sequence 10a, seed 42
- `tiny_20c_s5.yaml`: TinyImageNet dataset, sequence 20c, seed 5

## Key Configuration Parameters

Each configuration file includes the following important parameters:

1. `batch_size`: Size of the batch for training
2. `dataset`: Name of the dataset (e.g., five_datasets, split_cifar100, permuted_mnist, split_tinyimagenet)
3. `gpu`: GPU device number to use for training
4. `instruction_str`: Sequence of learning (L) and unlearning (U) instructions
5. `n_tasks`: Number of tasks in the continual learning setup
6. `resnet_type`: Type of ResNet architecture to use (e.g., 18, 50)
7. `seed`: Random seed for reproducibility

## Usage

To use a specific configuration file, pass its name (without the .yaml extension) as an argument when running the main script. For example:

This will load the configuration from `config/main/c100_10b_s5.yaml` and run the experiment with those settings.
