import os

from tqdm import tqdm

from find_ne import find_mixed_nash_equilibria, find_pure_nash_equilibria
from utils import (
    create_payoff_matrices,
    format_strategy,
    is_duplicate_ne,
    parse_nfg,
    pure_strategy_to_one_hot,
)


def nash(in_path: str, out_path: str):
    # Parse input file
    players, actions, payoffs = parse_nfg(in_path)
    payoff_matrices = create_payoff_matrices(players, actions, payoffs)

    # Find pure strategy NE
    pure_equilibria = find_pure_nash_equilibria(payoff_matrices)
    pure_equilibria = [pure_strategy_to_one_hot(eq, actions) for eq in pure_equilibria]

    # Find mixed strategy NE
    mixed_equilibria = find_mixed_nash_equilibria(payoff_matrices)

    # Remove duplicates
    all_equilibria = pure_equilibria
    for eq in mixed_equilibria:
        if is_duplicate_ne(eq, all_equilibria):
            continue
        all_equilibria.append(eq)

    with open(out_path, "w") as f:
        for eq in all_equilibria:
            strategies = []
            for strategy in eq:
                strategies.append(format_strategy(strategy))
            f.write(",".join(strategies) + "\n")


if __name__ == "__main__":
    for f in tqdm(os.listdir("input")):
        if f.endswith(".nfg"):
            nash("input/" + f, "output/" + f.replace("nfg", "ne"))

    # for f in os.listdir("examples"):
    #     if f.endswith(".nfg"):
    #         nash("examples/" + f, f.replace("nfg", "ne"))
