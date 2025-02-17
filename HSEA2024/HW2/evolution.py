import numpy as np


class Evolution:
    def __init__(self, args):
        self.args = args

        self.sigma = args.sigma
        self.p_crossover = args.p_crossover

    def bit_wise_mutation(self, parent):
        offspring = parent.copy()
        mutation_mask = np.random.rand(parent.shape[0], parent.shape[1]) < self.sigma
        offspring[mutation_mask] = 1 - offspring[mutation_mask]
        return offspring

    def uniform_crossover(self, parent):
        # devide parent
        np.random.shuffle(parent)
        couples = parent.reshape(-1, 2, parent.shape[-1])
        parent1 = couples[:, 0, :].reshape(-1, 1, parent.shape[-1])
        parent2 = couples[:, 1, :].reshape(-1, 1, parent.shape[-1])

        # random mask for crossover
        mask = np.random.rand(couples.shape[0], 1, couples.shape[2]) < self.p_crossover

        offspring1 = np.where(mask, parent1, parent2)
        offspring2 = np.where(mask, parent2, parent1)
        offspring = np.concatenate((offspring1, offspring2), axis=0).reshape(parent.shape)

        return offspring

    def heavy_tailed_mutation(self, parent):
        p, n = parent.shape

        for i in range(p):
            # Draw the number of bits to mutate based on a Cauchy distribution
            num_mutate = np.abs(np.random.standard_cauchy()) * n * self.sigma
            num_mutate = np.clip(num_mutate.astype(int), 0, n)

            # Select random indices in the vector to mutate
            mutation_indices = np.random.choice(n, num_mutate, replace=False)

            # Flip the selected bits
            offspring = parent.copy()
            offspring[i, mutation_indices] = 1 - offspring[i, mutation_indices]

        return offspring

    # def heavy_tailed_mutation(self, parent):
    #     offspring = parent.copy()
    #     mutation_mask = np.random.rand(parent.shape[0], parent.shape[1]) < self.sigma

    #     # Apply a heavy-tailed distribution (Cauchy distribution) to mutate values
    #     heavy_tailed_noise = np.random.standard_cauchy(size=parent.shape)

    #     # Normalize the noise and make it binary: flip bits with significant noise (positive or negative)
    #     normalized_noise = np.sign(heavy_tailed_noise)

    #     # Apply the mutation: flip the bits where mutation mask is True
    #     offspring[mutation_mask] = np.clip(offspring[mutation_mask] + normalized_noise[mutation_mask], 0, 1)

    #     return offspring
