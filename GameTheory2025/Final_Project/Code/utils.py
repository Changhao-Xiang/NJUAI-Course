import numpy as np


def parse_nfg(file_path: str) -> tuple[list[str], list[int], list[int]]:
    with open(file_path, "r") as f:
        content = f.read().strip()

    lines = content.split("\n")
    header_line = lines[0]

    # Find players
    players_start = content.find("{")
    players_end = content.find("}", players_start)
    players_str = content[players_start + 1 : players_end]

    # Extract player names
    players = []
    temp = players_str.strip()
    while '"' in temp:
        start = temp.find('"')
        end = temp.find('"', start + 1)
        players.append(temp[start + 1 : end])
        temp = temp[end + 1 :]

    # Find action numbers
    actions_start = content.find("{", players_end + 1)
    actions_end = content.find("}", actions_start)
    actions_str = content[actions_start + 1 : actions_end].strip()
    actions = [int(x) for x in actions_str.split()]

    # Find payoff data
    payoff_line = ""
    for line in lines:
        line = line.strip()
        if line and all(c.isdigit() or c.isspace() or c == "-" for c in line):
            try:
                test_payoffs = [int(x) for x in line.split()]
                payoff_line = line
                break
            except ValueError:
                continue

    payoffs = [int(x) for x in payoff_line.split()]

    return players, actions, payoffs


def create_payoff_matrices(players: list[str], actions: list[int], payoffs: list[int]) -> list[np.ndarray]:
    """
    Args:
        players: list of str, each is a player's name
        actions: list of int, each is the number of actions of a player
        payoffs: list of int, each is a payoff of a player
    Returns:
        list of numpy.ndarray, each is a payoff matrix of a player
    """
    num_players = len(players)
    total_outcomes = np.prod(actions).item()
    assert len(players) * total_outcomes == len(payoffs)

    # Create payoff matrices for each player
    payoff_matrices = []

    for player in range(num_players):
        # Create the payoff matrix of the player, shape is a tuple of actions
        payoff_matrix = np.zeros(actions, dtype=int)

        for outcome_idx in range(total_outcomes):
            # Convert linear index to multi-dimensional index (strategy combination)
            strategy_profile = []
            temp_idx = outcome_idx

            # The first player's action changes the fastest
            for action_idx in range(len(actions)):
                strategy_profile.append(temp_idx % actions[action_idx])
                temp_idx //= actions[action_idx]

            # Get the payoff of the player for the strategy combination
            payoff_matrix[tuple(strategy_profile)] = payoffs[outcome_idx * num_players + player]

        payoff_matrices.append(payoff_matrix)

    return payoff_matrices


def pure_strategy_to_one_hot(eq: list[int], actions: list[int]) -> list[list[float]]:
    """Convert a pure strategy to a onehot strategy representation"""
    onehot = []
    for i, action in enumerate(eq):
        strategy = [0] * actions[i]
        strategy[action] = 1
        onehot.append(strategy)
    return onehot


def format_strategy(strategy: list) -> str:
    """Format the strategy to the output format"""
    formatted = []
    for prob in strategy:
        assert isinstance(prob, (int, float))
        formatted.append(str(prob))
    return ",".join(formatted)


def is_duplicate_ne(new_ne: list[list[float]], cur_nes: list[list[list[float]]], tol: float = 1e-5) -> bool:
    """Check if the new NE is a duplicate of the existing NE"""
    if not cur_nes:
        return False
    for cur_ne in cur_nes:
        prob1 = np.concatenate(new_ne)
        prob2 = np.concatenate(cur_ne)
        if all(np.isclose(p1, p2, atol=tol) for p1, p2 in zip(prob1, prob2)):
            return True
    return False
