import itertools

import numpy as np


def find_pure_nash_equilibria(payoff_matrices: list[np.ndarray]) -> list[list[int]]:
    """
    Args:
        payoff_matrices: list of numpy.ndarray, each is a payoff matrix of a player
    Returns:
        equilibria: list of lists, each is a pure strategy profile of a Nash equilibrium
    """
    num_players = len(payoff_matrices)
    actions = payoff_matrices[0].shape
    equilibria = []

    # Iterate over all possible pure strategy combinations
    for strategy_profile in itertools.product(*[range(a) for a in actions]):
        is_equilibrium = True

        # Check if each player has a deviation incentive
        for player in range(num_players):
            current_payoff = payoff_matrices[player][strategy_profile]

            # Check if the player can get a higher payoff by changing their strategy
            for alt_action in range(actions[player]):
                if alt_action != strategy_profile[player]:
                    alt_profile = list(strategy_profile)
                    alt_profile[player] = alt_action
                    alt_payoff = payoff_matrices[player][tuple(alt_profile)]

                    if alt_payoff > current_payoff:
                        is_equilibrium = False
                        break

            if not is_equilibrium:
                break

        if is_equilibrium:
            equilibria.append(list(strategy_profile))

    return equilibria


def find_mixed_nash_equilibria(payoff_matrices: list[np.ndarray], tol: float = 1e-9) -> list[list[list[float]]]:
    """
    Args:
        payoff_matrices: list of numpy.ndarray, each is a payoff matrix of a player
        tol: float, numerical tolerance for comparisons
    Returns:
        list of tuples, each is a mixed strategy profile of a Nash equilibrium
    """
    num_players = len(payoff_matrices)
    actions = payoff_matrices[0].shape
    equilibria = []

    if num_players != 2:
        return equilibria

    A = payoff_matrices[0]  # Payoffs for Player 0
    B = payoff_matrices[1]  # Payoffs for Player 1
    m, n = actions

    # Enumerate supports for both players (non-empty subsets of actions)
    for size_s1 in range(1, m + 1):
        for S1 in itertools.combinations(range(m), size_s1):
            for size_s2 in range(1, n + 1):
                for S2 in itertools.combinations(range(n), size_s2):
                    # Define unknowns: probabilities for actions in S1 (p) and actions in S2 (q)
                    s1 = len(S1)
                    s2 = len(S2)

                    # Build linear system M * x = b, where x = [p_S1, q_S2]
                    num_vars = s1 + s2

                    # Equations:
                    # 1) Sum p = 1
                    # 2) Sum q = 1
                    # 3) For each j in S1 \ {j0}:   (A[j] ‑ A[j0]) · q = 0
                    # 4) For each k in S2 \ {k0}:   p · (B[:, k] ‑ B[:, k0]) = 0
                    rows = []
                    rhs = []

                    # Equation 1: sum p = 1
                    row = np.zeros(num_vars)
                    row[0:s1] = 1
                    rows.append(row)
                    rhs.append(1)

                    # Equation 2: sum q = 1
                    row = np.zeros(num_vars)
                    row[s1:] = 1
                    rows.append(row)
                    rhs.append(1)

                    # Choose reference actions
                    j0 = S1[0]
                    k0 = S2[0]

                    # Player 0 equal-payoff constraints
                    for j in S1[1:]:
                        row = np.zeros(num_vars)
                        # Only q variables appear in this constraint
                        for idx_q, k in enumerate(S2):
                            row[s1 + idx_q] = A[j, k] - A[j0, k]
                        rows.append(row)
                        rhs.append(0)

                    # Player 1 equal-payoff constraints
                    for k in S2[1:]:
                        row = np.zeros(num_vars)
                        for idx_p, j in enumerate(S1):
                            row[idx_p] = B[j, k] - B[j, k0]
                        rows.append(row)
                        rhs.append(0)

                    M = np.vstack(rows)
                    b_vec = np.array(rhs)

                    # Solve the linear system
                    try:
                        sol, residuals, rank, _ = np.linalg.lstsq(M, b_vec, rcond=None)
                    except np.linalg.LinAlgError:
                        continue

                    # Numerical clean-up
                    sol[np.abs(sol) < tol] = 0.0

                    # Split p and q
                    p = sol[:s1]
                    q = sol[s1:]

                    # Check feasibility: probabilities non-negative and sum to 1 within tolerance
                    if (p < -tol).any() or (q < -tol).any():
                        continue
                    p = np.maximum(p, 0)
                    q = np.maximum(q, 0)
                    if not (np.abs(p.sum() - 1) < 1e-6 and np.abs(q.sum() - 1) < 1e-6):
                        continue

                    # Ensure all support actions yield same payoff
                    # Player 0
                    u1_support = np.dot(A[j0, list(S2)], q)  # payoff of reference action j0
                    is_all_equal = True
                    for idx, j in enumerate(S1):
                        val = np.dot(A[j, list(S2)], q)
                        if np.abs(val - u1_support) > 1e-6:
                            is_all_equal = False
                            break
                    if not is_all_equal:
                        continue

                    # Player 1
                    u2_support = np.dot(p, B[np.array(S1), k0])  # payoff of reference action k0
                    for idx, k in enumerate(S2):
                        val = np.dot(p, B[np.array(S1), k])
                        if np.abs(val - u2_support) > 1e-6:
                            is_all_equal = False
                            break
                    if not is_all_equal:
                        continue

                    # Check no profitable deviation outside support
                    # Player 0 deviations
                    is_deviation_profitable = True
                    for j in set(range(m)) - set(S1):
                        if (A[j, :] @ probs_from_support(list(S2), q, n)) > u1_support + 1e-8:
                            is_deviation_profitable = False
                            break
                    if not is_deviation_profitable:
                        continue

                    # Player 1 deviations
                    for k in set(range(n)) - set(S2):
                        if np.dot(probs_from_support(list(S1), p, m), B[:, k]) > u2_support + 1e-8:
                            is_deviation_profitable = False
                            break
                    if not is_deviation_profitable:
                        continue

                    # Build full probability vectors
                    p_full = probs_from_support(list(S1), p, m)
                    q_full = probs_from_support(list(S2), q, n)

                    eq_profile = [p_full, q_full]
                    equilibria.append(eq_profile)

    return equilibria


def probs_from_support(support_indices: list[int], values: np.ndarray, total_actions: int) -> list[float]:
    full = np.zeros(total_actions)
    for idx, act in enumerate(support_indices):
        full[act] = values[idx]
    return full.tolist()