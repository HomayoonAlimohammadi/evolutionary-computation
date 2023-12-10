import random
from typing import List
import logging
import numpy as np
from enum import Enum
from tqdm import tqdm
import matplotlib.pyplot as plt


class VariationOperator(Enum):
    MUTATION_SCRAMBLE = 1
    MUTATION_INVERSION = 1
    CROSSOVER = 2


class MutationType(Enum):
    INVERSION = 1
    SCRAMBLE = 2


log_level = logging.INFO
logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger(__name__).setLevel(log_level)


class TSP:
    def __init__(self, dists_matrix, n_population: int):
        self.__best_costs: List[int] = []
        self.__worst_costs: List[int] = []
        self.__dists_matrix = dists_matrix
        assert len(dists_matrix) == len(dists_matrix[0])
        self.__n_cities = len(dists_matrix)

        self.__population = np.zeros((n_population, self.__n_cities), dtype=np.int16)
        for i in tqdm(range(n_population), desc="Initializing Population"):
            self.__population[i] = self.__get_random_sample()

    def __get_random_sample(self):
        s = np.arange(self.__n_cities)
        np.random.shuffle(s)
        return s

    def __cost(self, s):
        cost = 0
        for i in range(len(s)):
            a, b = s[i], s[i - 1]
            cost += self.__dist(a, b)

        return cost

    def __dist(self, a, b):
        return self.__dists_matrix[a, b]

    def __mutate(self, s, typ: MutationType):
        x, y = random.sample(range(self.__n_cities), 2)
        first, last = min(x, y), max(x, y)
        subset = np.copy(s[first:last])
        if typ == MutationType.INVERSION:
            return list(s[:first]) + list(subset[::-1]) + list(s[last:])
        elif typ == MutationType.SCRAMBLE:
            np.random.shuffle(subset)
            return list(s[:first]) + list(subset) + list(s[last:])
        else:
            raise ValueError(f"invalid mutation type {typ}")

    def __crossover(self, s1, s2):
        assert len(s1) == len(s2)
        co_point = random.randint(1, len(s1) - 1)
        return s1[:co_point].copy() + s2[co_point:].copy()

    def __pmx(self, s1, s2):
        raise NotImplementedError

    def iterate(self, n_iters: int, var_op: VariationOperator):
        for _ in tqdm(range(n_iters), desc="Iterating"):
            if var_op == VariationOperator.MUTATION_INVERSION:
                self.__population = self.__mutate_population(
                    self.__population,
                    MutationType.INVERSION,
                    len(self.__population),
                )
            if var_op == VariationOperator.MUTATION_SCRAMBLE:
                self.__population = self.__mutate_population(
                    self.__population,
                    MutationType.SCRAMBLE,
                    len(self.__population),
                )
            elif var_op == VariationOperator.CROSSOVER:
                self.__population = self.__crossover_population(
                    self.__population, len(self.__population)
                )
            else:
                raise ValueError(f"invalid variation operator {var_op}")

            self.__best_route, best_cost = self.__find_best(self.__cost)
            self.__worst_route, worst_cost = self.__find_worst(self.__cost)
            self.__best_costs.append(best_cost)
            self.__worst_costs.append(worst_cost)

    def __mutate_population(self, init_pop, typ: MutationType, final_pop_size: int):
        new_pop = [self.__mutate(p, typ=typ) for p in init_pop]
        new_pop.extend(init_pop)
        return self.__select(new_pop, final_pop_size)

    def __crossover_population(self, init_pop, final_pop_size: int):
        new_pop = init_pop.copy()
        for _ in range(len(init_pop)):
            couple = random.sample(init_pop, 2)
            new_pop.append(self.__crossover(couple[0], couple[1]))

        return self.__select(new_pop, final_pop_size)

    def __find_best(self, cost_fn):
        best_cost = cost_fn(self.__population[0])
        best = self.__population[0]
        for p in self.__population:
            cost = cost_fn(p)
            if cost < best_cost:
                best_cost = cost
                best = p

        return best, best_cost

    def __find_worst(self, cost_fn):
        worst_cost = cost_fn(self.__population[0])
        worst = self.__population[0]
        for p in self.__population:
            cost = cost_fn(p)
            if cost > worst_cost:
                worst_cost = cost
                worst = p

        return worst, worst_cost

    def __select(self, pop, pop_size: int):
        assert len(pop) >= pop_size

        fs = [(i, self.__cost(p)) for i, p in enumerate(pop)]
        fs = sorted(fs, key=lambda x: x[1])
        selected = []
        for i in range(pop_size):
            idx, _ = fs[i]
            selected.append(pop[idx])

        return selected

    def plot_costs(self):
        plt.plot(range(1, len(self.__best_costs) + 1), self.__best_costs, label="Best")
        plt.plot(
            range(1, len(self.__worst_costs) + 1), self.__worst_costs, label="Worst"
        )
        plt.xlabel("# Iteration")
        plt.ylabel("Score")
        plt.axhline(
            y=self.__best_costs[-1],
            color="green",
            linestyle="--",
            label=f"Final Best: {self.__best_costs[-1]}",
        )
        plt.axhline(
            y=self.__worst_costs[-1],
            color="red",
            linestyle="--",
            label=f"Final Worst: {self.__worst_costs[-1]}",
        )
        plt.axhline(
            y=self.__best_costs[0],
            color="green",
            linestyle="dotted",
            label=f"First Best: {self.__best_costs[0]}",
        )
        plt.axhline(
            y=self.__worst_costs[0],
            color="red",
            linestyle="dotted",
            label=f"First Worst: {self.__worst_costs[0]}",
        )
        plt.legend()
        plt.show()

    def plot_route(self):
        plt.legend()
        plt.show()


if __name__ == "__main__":
    with open("cities.txt") as f:
        lines = f.readlines()

    dists_matrix = np.zeros((30, 30))

    for i in range(2, 32):
        line = lines[i]
        line = line.replace("\n", "")
        dists = line.split("\t")
        dists_matrix[i - 2] = dists

    tsp = TSP(dists_matrix, n_population=100)
    tsp.iterate(1000, VariationOperator.MUTATION_INVERSION)
    tsp.plot_costs()
    # tsp.plot_route()
