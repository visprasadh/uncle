import os
import json
import numpy as np
import pandas as pd


def get_slices(sequence):
    slices, indexes, stack = {}, {}, {}
    cnt, len_seq = 0, len(sequence)
    for i, ele in enumerate(sequence):
        if ele[0] == "L":
            indexes[ele] = cnt
            cnt += 1
            stack[ele[1:]] = (ele, i)
        elif ele[0] == "U":
            indexes[ele] = indexes[ele.replace("U", "L")]
            if ele[1:] in stack:
                l_ele, l_index = stack.pop(ele[1:])
                slices[l_ele] = (l_index, i)
                slices[ele] = (i, len_seq)
            else:
                slices[ele] = (i, len_seq)

    # Add remaining 'L' elements in the stack
    for l_ele, l_index in stack.values():
        slices[l_ele] = (l_index, len_seq)

    return slices, indexes


def get_dataset_seq(name):
    dataset_mapping = {
        "c100": "cifar100",
        "5d": "five_datasets",
        "tiny": "tiny",
        "pm": "pmnist",
    }
    seq_mapping = {"a_": "seq1", "b_": "seq2", "c_": "seq3"}

    dataset = next(
        (dataset_mapping[key] for key in dataset_mapping if key in name), None
    )
    seq = next((seq_mapping[key] for key in seq_mapping if key in name), None)

    return dataset, seq


def calculate_stability(s, acc_matrix, name):
    dataset, seq = get_dataset_seq(name)
    # print(dataset, seq)
    sequence = s[dataset][seq].split(",")
    slices, indexes = get_slices(sequence)
    # print(slices)
    # print(indexes)

    df = pd.DataFrame(columns=["Task", "Stability", "Std"])
    for ele in sequence:
        s_idx, l_idx = slices[ele]
        array = acc_matrix[s_idx:l_idx, indexes[ele]]
        if ele[0] == "L":
            stability = np.mean((1 - ((array[0] - array) / array[0])) * 100)
            std = np.std(array - array[0])
        else:
            # stability = np.mean(array[0] - array)
            # std = np.std(array[0] - array)
            stability = np.mean(10.0 - array)
            std = np.std(10.0 - array)
        df.loc[len(df)] = [ele, round(stability, 2), round(std, 2)]
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculate stability for given path.")
    parser.add_argument("--path", type=str, help="Path to the logs directory")
    args = parser.parse_args()

    PATH = args.path
    sequences = json.load(open("./json/sequences.json"))

    list_dir = os.listdir(PATH)
    list_dir.sort()

    for dir in list_dir:
        print(dir)
        acc_matrix = np.loadtxt(
            os.path.join(PATH, dir, "acc_matrix.csv"), delimiter=","
        )
        stability_df = calculate_stability(sequences, acc_matrix, dir)
        # stability_df.to_csv(os.path.join(PATH, dir, "stability.csv"))
        stability_df.to_csv(os.path.join(PATH, dir, "stability_5p0.csv"))
