import argparse
import csv
import datetime
import os
import random

import numpy as np
import yaml
from mutation import CreepMutation, ResetMutation, SwapMutation
from place_db import PlaceDB
from utils import evaluate, rank_macros, write_final_placement


class BBOPlacer:
    """Basic class for WireMask-BBO"""

    def __init__(
        self, dim, grid_num, grid_size, placedb, node_id_ls, csv_writer, csv_file, placement_save_dir
    ):
        self.dim = dim
        self.lb = 0 * np.ones(dim)
        self.ub = grid_num * np.ones(dim)
        self.grid_num = grid_num
        self.grid_size = grid_size
        self.placedb = placedb
        self.node_id_ls = node_id_ls
        self.csv_writer = csv_writer
        self.csv_file = csv_file
        self.best_hpwl = 1e12
        self.placement_save_dir = placement_save_dir

    def _evaluate(self, x):
        """
        Evaluate by WireMask-BBO

        Returns:
            hpwl value of solution x
        """
        assert len(x) == self.dim
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        place_record = {}
        node_id_ls = self.node_id_ls.copy()
        for i in range(len(node_id_ls)):
            place_record[node_id_ls[i]] = {}
            place_record[node_id_ls[i]]["loc_x"] = x[i * 2]
            place_record[node_id_ls[i]]["loc_y"] = x[i * 2 + 1]
        placed_macro, hpwl = evaluate(
            self.node_id_ls, self.placedb, self.grid_num, self.grid_size, place_record
        )
        if hpwl < self.best_hpwl:
            self.best_hpwl = hpwl
            write_final_placement(placed_macro, self.placement_save_dir)
        self.csv_writer.writerow([hpwl])
        self.csv_file.flush()
        return hpwl


class RandomSearch:
    """Simple implementation of Random search."""

    def __init__(self, placer, max_iteration: int = 1000):
        """
        Initialize the RandomSearch object.

        Args:
            placer: An object representing the macro placer.
            max_iteration: Maximum number of iterations (default: 1000).
        """
        self.placer = placer
        self.max_iteration = max_iteration

    def init(self):
        """Initialize the search by generating a random solution."""
        self.x = np.random.randint(self.placer.lb, self.placer.ub + 1, self.placer.dim)

    def step(self):
        """Take a step in the search by generating a new random solution. RS just randomly generate a new solution"""
        self.init()

    def evaluate(self):
        """Evaluate the current solution."""
        return self.placer._evaluate(self.x)

    def run(self):
        """Run the random search algorithm."""
        self.init()
        for i in range(self.max_iteration):
            value = self.evaluate()
            print(f"HPWL at iteration {i} is {value}")
            self.step()


class EvolutionAlgorithm:
    """Simple implementation of (1+1) evolution algorithm."""

    def __init__(self, placer, max_iteration: int = 1000, mutation: str = "swap"):
        """
        Initialize the EvolutionAlgorithm object.

        Args:
            placer: An object representing the macro placer.
            max_iteration: Maximum number of iterations (default: 1000).
        """
        self.placer = placer
        self.max_iteration = max_iteration

        # Initialize mutation operators
        self.mutations = {
            "swap": SwapMutation(),
            "reset": ResetMutation(self.placer.lb, self.placer.ub),
            "creep": CreepMutation(self.placer.lb, self.placer.ub),
        }
        self.mutation = self.mutations[mutation]

    def init(self):
        """Initialize the search by generating 100 random solutions and pick the best one."""
        print("Initializing population...")
        init_population = np.random.randint(self.placer.lb, self.placer.ub + 1, size=(100, self.placer.dim))
        fitness_values = [self.placer._evaluate(ind) for ind in init_population]
        best_index = np.argmin(fitness_values)
        self.x = init_population[best_index]

    def step(self):
        """Take a step in the search by generating a new random solution. (1+1) EA randomly exchange two macros' locations"""
        new_x = self.mutation(self.x)

        current_value = self.placer._evaluate(self.x)
        new_value = self.placer._evaluate(new_x)

        if new_value < current_value:
            self.x = new_x

    def evaluate(self):
        """Evaluate the current solution."""
        return self.placer._evaluate(self.x)

    def run(self):
        """Run the random search algorithm."""
        self.init()
        for i in range(self.max_iteration):
            value = self.evaluate()
            print(f"HPWL at iteration {i} is {value}")
            self.step()


class PopulationEvolutionAlgorithm:
    """Improved implementation of (μ+λ) evolution algorithm."""

    def __init__(
        self,
        placer,
        max_iteration: int = 1000,
        mutation: str = "swap",
        population_size: int = 20,
        offspring_size: int = 10,
    ):
        """
        Initialize the EvolutionAlgorithm object.

        Args:
            placer: An object representing the macro placer.
            max_iteration: Maximum number of iterations (default: 1000).
            mutation: Mutation type (swap/reset/creep).
            population_size: Number of parent individuals (μ).
            offspring_size: Number of offspring individuals (λ).
        """
        self.placer = placer
        self.max_iteration = max_iteration
        self.population_size = population_size
        self.offspring_size = offspring_size

        # Initialize mutation operators
        self.mutations = {
            "swap": SwapMutation(),
            "reset": ResetMutation(self.placer.lb, self.placer.ub),
            "creep": CreepMutation(self.placer.lb, self.placer.ub),
        }
        self.mutation = self.mutations[mutation]

    def init(self):
        """Initialize the population with random solutions."""
        print("Initializing population...")
        self.population = np.random.randint(
            self.placer.lb, self.placer.ub + 1, size=(self.population_size, self.placer.dim)
        )
        # Evaluate initial population
        self.fitness_values = np.array([self.placer._evaluate(ind) for ind in self.population])

    def tournament_selection(self, tournament_size=3):
        """Select individuals using tournament selection."""
        selected = []
        for _ in range(self.offspring_size):
            # Randomly select tournament_size individuals
            candidates = np.random.choice(len(self.population), tournament_size, replace=False)
            # Select the best one
            winner = candidates[np.argmin(self.fitness_values[candidates])]
            selected.append(self.population[winner])
        return np.array(selected)

    def generate_offspring(self, parents):
        """Generate offspring population through mutation."""
        offspring = []
        for parent in parents:
            # Perform crossover with 50% probability
            if np.random.rand() < 0.5:
                # Select another random parent
                other_parent = parents[np.random.randint(len(parents))]
                # Choose random crossover point
                crossover_point = np.random.randint(1, len(parent) - 1)
                # Create child by combining parts of both parents
                child = np.concatenate([parent[:crossover_point], other_parent[crossover_point:]])
                parent = child  # Use the crossover result for mutation

            # Apply mutation to create new offspring
            child = self.mutation(parent)
            offspring.append(child)
        return np.array(offspring)

    def step(self):
        """Perform one generation of evolution."""
        parents = self.tournament_selection()
        offspring = self.generate_offspring(parents)

        # Evaluate offspring
        offspring_fitness = np.array([self.placer._evaluate(ind) for ind in offspring])

        # Combine parents and offspring
        combined_population = np.vstack([self.population, offspring])
        combined_fitness = np.hstack([self.fitness_values, offspring_fitness])

        # Select new population
        best_indices = np.argpartition(combined_fitness, self.population_size)[: self.population_size]
        self.population = combined_population[best_indices]
        self.fitness_values = combined_fitness[best_indices]

    def evaluate(self):
        """Evaluate the best solution in current population."""
        return np.min(self.fitness_values)

    def run(self):
        """Run the evolution algorithm."""
        self.init()
        for i in range(self.max_iteration):
            value = self.evaluate()
            print(f"HPWL at iteration {i} is {value}")
            self.step()


def main(args):
    current_time = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
    timestamp = current_time.strftime("%Y%m%d_%H%M%S")
    dataset = args.dataset
    random.seed(args.seed)
    np.random.seed(args.seed)
    with open("../config/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    method = config["method"]
    mutation = args.mutation

    # read data_set
    placedb = PlaceDB(dataset)
    node_id_ls = rank_macros(placedb, rank_key="area")
    grid_setting = config["grid_setting"]
    grid_num = grid_setting[dataset]["grid_num"]
    grid_size = grid_setting[dataset]["grid_size"]
    macro_num = len(placedb.node_info.keys())
    dim = 2 * macro_num
    # save data directory
    hpwl_save_dir = f"./result/{method}_{mutation}/curve/"
    placement_result_save_dir = f"./result/{method}_{mutation}/placement_result/"
    if not os.path.exists(hpwl_save_dir):
        os.makedirs(hpwl_save_dir)
    if not os.path.exists(placement_result_save_dir):
        os.makedirs(placement_result_save_dir)
    if args.timestamp:
        hpwl_save_dir += f"{dataset}_{args.seed}_{timestamp}.csv"
        placement_result_save_dir += f"{dataset}_{args.seed}_{timestamp}.csv"
    else:
        hpwl_save_dir += f"{dataset}_{args.seed}.csv"
        placement_result_save_dir += f"{dataset}_{args.seed}.csv"
    hpwl_save_file = open(hpwl_save_dir, "a+")
    hpwl_writer = csv.writer(hpwl_save_file)
    print(f"Running {method} on {args.dataset} with seed {args.seed}")
    print(f"HPWL log is {hpwl_save_dir}")
    # Run
    bbo_placer = BBOPlacer(
        dim=dim,
        grid_num=grid_num,
        grid_size=grid_size,
        placedb=placedb,
        node_id_ls=node_id_ls,
        csv_writer=hpwl_writer,
        csv_file=hpwl_save_file,
        placement_save_dir=placement_result_save_dir,
    )
    method_map = {
        "random_search": RandomSearch(placer=bbo_placer, max_iteration=args.max_iteration),
        "ea": EvolutionAlgorithm(placer=bbo_placer, max_iteration=args.max_iteration, mutation=mutation),
        "ea_with_population": PopulationEvolutionAlgorithm(
            placer=bbo_placer,
            max_iteration=args.max_iteration,
            mutation=mutation,
            population_size=20,
            offspring_size=10,
        ),
    }
    algo = method_map[method]
    algo.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="adaptec1")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--max_iteration", type=int, default=200)
    parser.add_argument("--timestamp", action="store_true", help="If use the timestamp in name")
    parser.add_argument("--mutation", type=str, default="swap")
    args = parser.parse_known_args()[0]
    main(args=args)
