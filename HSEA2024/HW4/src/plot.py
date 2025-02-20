import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

maskpalce_results = {
    "adaptec1": 6.38,
    "adaptec2": 73,
    "adaptec3": 84,
    "adaptec4": 79,
    "bigblue1": 2.39,
    "bigblue3": 91,
}
# Results of MaskPlace

ylim = {
    "adaptec1": [5.7, 9],
    "adaptec2": [45, 90],
    "adaptec3": [55, 90],
    "adaptec4": [56, 84],
    "bigblue1": [2.1, 2.45],
    "bigblue3": [55, 105],
}

color_set = {
    "Amaranth": np.array([0.9, 0.17, 0.31]),  # main algo
    "Amber": np.array([1.0, 0.49, 0.0]),  # main baseline
    "Bleu de France": np.array([0.19, 0.55, 0.91]),
    "Electric violet": np.array([0.56, 0.0, 1.0]),
    "Arsenic": np.array([0.23, 0.27, 0.29]),
    "Blush": np.array([0.87, 0.36, 0.51]),
    "Dark sea green": np.array([0.56, 0.74, 0.56]),
    "Dark electric blue": np.array([0.33, 0.41, 0.47]),
    "Dark gray": np.array([0.66, 0.66, 0.66]),
    "French beige": np.array([0.65, 0.48, 0.36]),
    "Grullo": np.array([0.66, 0.6, 0.53]),
    "Dark coral": np.array([0.8, 0.36, 0.27]),
    "Old lavender": np.array([0.47, 0.41, 0.47]),
    "Sandy brown": np.array([0.96, 0.64, 0.38]),
    "Dark cyan": np.array([0.0, 0.55, 0.55]),
    "Brick red": np.array([0.8, 0.25, 0.33]),
    "Dark pastel green": np.array([0.01, 0.75, 0.24]),
}

with open("../config/config.yaml", "r") as file:
    config = yaml.safe_load(file)
method = config["method"]


def plot_curves(dataset="adaptec1"):
    # seeds = [2023, 2024, 2025]
    seeds = [2024]
    mutation = "swap"
    fig, ax = plt.subplots()
    # Draw maskplace baseline
    plt.axhline(y=maskpalce_results[dataset], ls=":", color="black", label="MaskPlace", linewidth=2.5)
    all_data = pd.DataFrame()
    # Draw Random curve
    min_length = float("inf")
    for seed in seeds:
        # dir = f"result/{method}/curve/{dataset}_{seed}.csv"
        dir = f"result/{method}_{mutation}/curve/{dataset}_{seed}.csv"
        data = pd.read_csv(dir, header=None).astype(float)
        all_data[f"Seed {seed}"] = data
        min_length = min(min_length, len(data))
    all_data = all_data.iloc[:min_length, :]

    mean = all_data.mean(axis=1)
    std = all_data.std(axis=1)
    mean_np = np.minimum.accumulate(mean.to_numpy()) / 1e5
    std_np = np.maximum.accumulate(std.to_numpy()) / 1e5

    # plt.figure(figsize=(10, 6))
    ax.plot(mean_np, color=color_set["Amaranth"], label=method)
    ax.fill_between(mean.index, mean_np - std_np, mean_np + std_np, color=color_set["Amaranth"], alpha=0.5)
    ax.set_ylim(ylim[dataset][0], ylim[dataset][1])
    ax.set_xlabel("Iterations", fontsize=15)
    ax.set_ylabel("HPWL (*1e5)", fontsize=15)
    ax.set_title(dataset, fontsize=17)
    plt.subplots_adjust(bottom=0.12)
    plt.legend(fontsize=15)
    plt.savefig(f"result/{method}_{mutation}_{dataset}.pdf", dpi=1000, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="adaptec1")
    args = parser.parse_known_args()[0]
    plot_curves(dataset=args.dataset)
