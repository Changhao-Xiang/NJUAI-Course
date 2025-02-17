import argparse
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from evolution import Evolution


def generate_regular_graph(args):
    # 这里简单以正则图为例, 鼓励同学们尝试在其他类型的图(具体可查看如下的nx文档)上测试算法性能
    # nx文档 https://networkx.org/documentation/stable/reference/generators.html
    graph = nx.random_graphs.random_regular_graph(d=args.n_d, n=args.n_nodes, seed=args.seed_g)
    return graph, len(graph.nodes), len(graph.edges)


def generate_gset_graph(args):
    # 这里提供了比较流行的图集合: Gset, 用于进行分割
    dir = "./Gset/"
    fname = dir + "G" + str(args.gset_id) + ".txt"
    graph_file = open(fname)
    n_nodes, n_e = graph_file.readline().rstrip().split(" ")
    print(n_nodes, n_e)
    nodes = [i for i in range(int(n_nodes))]
    edges = []
    for line in graph_file:
        n1, n2, w = line.split(" ")
        edges.append((int(n1) - 1, int(n2) - 1, int(w)))
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_weighted_edges_from(edges)
    return graph, len(graph.nodes), len(graph.edges)


def graph_generator(args):
    if args.graph_type == "regular":
        return generate_regular_graph(args)
    elif args.graph_type == "gset":
        return generate_gset_graph(args)
    else:
        raise NotImplementedError(f"Wrong graph_tpye")


def get_fitness(graph, x, n_edges, threshold=0.5):
    x_eval = np.where(x >= threshold, 1, -1)

    g1 = np.where(x_eval == -1)[0]
    g2 = np.where(x_eval == 1)[0]
    return nx.cut_size(graph, g1, g2) / n_edges


def get_topk_fitness(topk, graph, x, n_edges, threshold=0.5):
    all_fitness = np.zeros((x.shape[0],))
    for i, solution in enumerate(x):
        all_fitness[i] = get_fitness(graph, solution, n_edges, threshold)

    sorted_indices = np.argsort(all_fitness)[::-1]

    top_k_indices = sorted_indices[:topk]
    top_k_values = all_fitness[top_k_indices]

    return top_k_values, top_k_indices


def evolve(args, evolution, evol_name, graph, init_parent, n_edges):
    graph_name = f"{args.graph_type}_{args.gset_id}"
    parent = init_parent.copy()

    best_fitness, _ = get_topk_fitness(1, graph, parent, n_edges)
    best_fitness = best_fitness[0]
    best_fitness_record = [best_fitness]
    for i in range(args.T):
        if "bit_wise_mutation" in evol_name:
            offspring = evolution.bit_wise_mutation(parent)

        elif "uniform_crossover" in evol_name:
            offspring = evolution.uniform_crossover(parent)

        elif "crossover_with_bit_wise" in evol_name:
            offspring = evolution.uniform_crossover(parent)
            offspring = evolution.bit_wise_mutation(offspring)

        elif "heavy_tailed_mutation" in evol_name:
            offspring = evolution.heavy_tailed_mutation(parent)

        elif "crossover_with_heavy_tailed" in evol_name:
            offspring = evolution.uniform_crossover(parent)
            offspring = evolution.heavy_tailed_mutation(offspring)

        # new population
        import pdb

        pdb.set_trace()
        cur_population = np.concatenate([parent, offspring])

        # update parent with top-k solution
        topk_fitness, topk_indices = get_topk_fitness(args.population_size, graph, cur_population, n_edges)
        parent = cur_population[topk_indices]

        # update the best fitness if a better solution is found
        if topk_fitness[0] > best_fitness:
            best_fitness = topk_fitness[0]
        best_fitness_record.append(best_fitness)

        if i % 1000 == 0:
            print(f"*****{graph_name}: {evol_name}*****")
            print(f"Generation {i+1}: Best Fitness = {best_fitness}")

    return best_fitness_record


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-type", type=str, help="graph type", default="gset")
    parser.add_argument("--n-nodes", type=int, help="the number of nodes", default=1000)
    parser.add_argument("--n-d", type=int, help="the number of degrees for each node", default=10)
    parser.add_argument("--T", type=int, help="the number of fitness evaluations", default=10000)
    parser.add_argument("--seed-g", type=int, help="the seed of generating regular graph", default=1)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--gset-id", type=int, default=1)
    parser.add_argument("--sigma", type=float, help="hyper-parameter of mutation operator", default=0.5)
    parser.add_argument("--population_size", type=int, help="the size of population", default=10)
    parser.add_argument("--p_crossover", type=int, help="the probability of uniform crossover", default=0.5)

    parser.add_argument("--heavy_tailed", type=bool, default=False)

    args = parser.parse_known_args()[0]
    return args


def main(args=get_args()):
    print(args)
    graph, n_nodes, n_edges = graph_generator(args)
    np.random.seed(args.seed)

    init_parent = np.random.randint(0, 2, (args.population_size, n_nodes))
    evolution = Evolution(args)

    graph_name = f"{args.graph_type}_{args.gset_id}"
    os.makedirs(f"./records/{graph_name}", exist_ok=True)

    if not args.heavy_tailed:
        evol_names = ["bit_wise_mutation", "uniform_crossover", "crossover_with_bit_wise"]
        # evol_names = ["uniform_crossover"]
    else:
        evol_names = ["crossover_with_bit_wise", "crossover_with_heavy_tailed"]
        # evol_names = ["heavy_tailed_mutation"]

    records = []
    for evol_name in evol_names:
        save_path = f"./records/{graph_name}/{evol_name}.txt"
        record = evolve(args, evolution, evol_name, graph, init_parent, n_edges)
        records.append(record)

        with open(save_path, "w") as f:
            for fitness in record:
                f.write(f"{fitness:.5f}\n")

    # plot results
    plt.figure()
    for record in records:
        plt.plot(record)

    plt.legend(evol_names)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("Best Fitness per Generation")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./records/{graph_name}/fitness.png")


if __name__ == "__main__":
    main()
