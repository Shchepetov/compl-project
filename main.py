import random
import math
import time

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class Population:
    def __init__(self, graph, population_size):

        self.graph = graph
        self.n_generation = 0
        self.genome_size = len(self.graph.nodes)
        self.individuals = self.random_population(population_size)
        self.min_distance = self.fittest_distance()

    def random_population(self, population_size):

        population_genomes = [
            self.random_genome() for _ in range(population_size)
        ]
        individuals = [
            Individual(self.graph, genome) for genome in population_genomes
        ]

        return individuals

    def random_genome(self):

        genome = list(self.graph.nodes)
        random.shuffle(genome)
        return genome

    def fit(
        self,
        population_size,
        elite_size,
        mutation_prob,
        epoch_limit=None,
        blank_epoch_limit=None,
        stop_limit=None,
    ):

        epoch = 0
        blank_epoch = 0
        fit = self.fittest_distance()
        while (
            (not epoch_limit or epoch < epoch_limit)
            and (not stop_limit or fit > stop_limit)
            and (not blank_epoch_limit or blank_epoch < blank_epoch_limit)
        ):

            self.next_generation(population_size, elite_size, mutation_prob)
            fit = self.fittest_distance()
            epoch += 1
            if fit >= self.min_distance:
                blank_epoch += 1
            else:
                self.min_distance = fit
                blank_epoch = 0

            yield fit

    def next_generation(self, population_size, elite_size, mutation_prob):

        assert (
            elite_size <= population_size
        ), "Elite should be less or equal population size"

        children = []
        for _ in range(population_size - elite_size):
            parent_1, parent_2 = random.choices(
                self.individuals,
                weights=[individual.fitness for individual in self.individuals],
                k=2,
            )
            child_genome = self.breed(parent_1, parent_2)
            if random.random() < mutation_prob:
                child_genome = self.mutation(child_genome)
            child = Individual(self.graph, child_genome)
            children.append(child)

        survivors = random.choices(
            self.individuals,
            weights=[individual.fitness for individual in self.individuals],
            k=elite_size,
        )

        new_generation = survivors + children

        self.individuals = new_generation
        self.n_generation += 1

    def breed(self, parent_1, parent_2):

        genome_1, genome_2 = parent_1.genome, parent_2.genome
        child_genome = []

        sep = random.randint(1, self.genome_size - 1)
        dominant, recessive = (
            (genome_1, genome_2)
            if random.random() < 0.5
            else (genome_2, genome_1)
        )
        prefix_genome, suffix_genome = dominant[:sep], dominant[sep:]
        child_genome += prefix_genome

        for chromosome in recessive[sep:]:
            if chromosome not in prefix_genome:
                child_genome.append(chromosome)

        while len(child_genome) < self.genome_size:
            chromosome = suffix_genome.pop()
            if chromosome not in child_genome:
                child_genome.append(chromosome)

        return child_genome

    def mutation(self, genome):

        idx = range(len(genome))
        i1, i2 = random.sample(idx, 2)
        genome[i1], genome[i2] = genome[i2], genome[i1]

        return genome

    def fittest_distance(self):

        return min(self.individuals).distance


class Individual:
    def __init__(self, graph, route):

        assert len(route) == len(graph.nodes)

        self.genome = route
        self.distance = nx.path_weight(graph, route + [route[0]], "length")
        self.fitness = (1 / self.distance) ** 2

    def __lt__(self, other):

        return self.fitness < other.fitness

    def __repr__(self):

        if len(self.genome) < 5:

            return repr(self.genome)

        prefix = [str(vert) for vert in self.genome[:2]]
        suffix = [str(vert) for vert in self.genome[-2:]]

        return (
            f"[{', '.join(prefix)}, ..., {', '.join(suffix)}]: {self.fitness}"
        )


def show(data, ms):

    plt.figure(figsize=(10, 5))
    plt.plot(data)
    annotate_absolute_min(data)
    plt.xlabel("Эпоха")
    plt.ylabel("Минимальный путь")
    print(f"In {1000*ms} ms.")


def annotate_absolute_min(data):

    xmin = np.argmin(data)
    ymin = data.min()
    text = "min: {} ({} эпоха)".format(ymin, xmin)
    ax = plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops = dict(
        arrowstyle="->", connectionstyle="angle,angleA=0,angleB=60"
    )
    kw = dict(
        xycoords="data",
        textcoords="axes fraction",
        arrowprops=arrowprops,
        bbox=bbox_props,
        ha="right",
        va="top",
    )
    ax.annotate(text, xy=(xmin, ymin), xytext=(0.94, 0.96), **kw)


def comparative_data(graph, weight="length"):

    algs_results = {}

    init_cycle = list(graph.nodes)
    random.shuffle(init_cycle)
    init_cycle.append(init_cycle[0])

    annealing_path, annealing_time = measure_time(
        nx.algorithms.approximation.traveling_salesman.simulated_annealing_tsp,
        graph,
        weight=weight,
        init_cycle=init_cycle,
    )
    christ_path, christ_time = measure_time(
        nx.algorithms.approximation.traveling_salesman.christofides,
        graph,
        weight="length",
    )
    greedy_path, greedy_time = measure_time(
        nx.algorithms.approximation.traveling_salesman.greedy_tsp,
        graph,
        weight="length",
    )

    algs_results = {
        "annealing": {
            "path": annealing_path,
            weight: nx.path_weight(graph, annealing_path, weight),
            "time": annealing_time,
        },
        "christ": {
            "path": christ_path,
            weight: nx.path_weight(graph, christ_path, weight),
            "time": christ_time,
        },
        "greedy": {
            "path": greedy_path,
            weight: nx.path_weight(graph, greedy_path, weight),
            "time": greedy_time,
        },
    }

    return algs_results


def measure_time(f, *args, **kwargs):

    t1 = time.perf_counter()
    result = f(*args, **kwargs)
    t2 = time.perf_counter()

    return result, t2 - t1


GRAPH_SIZE = 50
MAX_WEIGHT = 5

graph = nx.fast_gnp_random_graph(GRAPH_SIZE, 1)
for (u, v) in graph.edges():
    graph.edges[u, v]["length"] = random.randint(1, MAX_WEIGHT)
nx.draw(graph)

other_algs_results = comparative_data(graph)

POP_SIZE = 150
ELITE_SIZE = 135
MUTATION_PROB = 0.15
BLANK_EPOCH_LIMIT = 5000

population = Population(graph, POP_SIZE)
init_history = pd.Series([population.fittest_distance()])

t1 = time.perf_counter()
history = init_history.append(
    pd.Series(
        population.fit(
            POP_SIZE,
            ELITE_SIZE,
            MUTATION_PROB,
            blank_epoch_limit=BLANK_EPOCH_LIMIT,
        )
    ),
    ignore_index=True,
)
t2 = time.perf_counter()

show(history, t2 - t1)
