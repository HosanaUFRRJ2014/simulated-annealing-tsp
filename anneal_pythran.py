# pythran export anneal_cythran(float list, float, float, int, float, int, int, float list, float, float list, float, int list, float list)
# pythran export fitness_function(float list, int, int list)

import random
import numpy as np



def p_accept_function(candidate_fitness, cur_fitness, T):
    """
    Probability of accepting if the candidate is worse than current.
    Depends on the current temperature and difference between candidate and current.
    """
    return np.exp(-abs(candidate_fitness - cur_fitness) / T)


def fitness_function(candidate, N, coords):
    """
    Total distance of the current solution path.
    """
    cur_fit = 0.0
    i = 0
    node_0 = 0
    node_1 = 0
    coord_0 = 0
    coord_1 = 0
    x = 0
    y = 0
    c = 0.0
    while i < N:
        index_0 = i % N
        index_1 = (i + 1) % N
        node_0 = candidate[index_0]
        node_1 = candidate[index_1]
        coord_0 = coords[node_0]
        coord_1 = coords[node_1]
        x = coord_0[0] - coord_1[0]
        y = coord_0[0] - coord_1[0]
        z = x**2 + y**2
        # ** reparem que se trocar a linha abaixo por c = 1 funciona
        c = np.sqrt(z)
        cur_fit = cur_fit + c
        i += 1
    return cur_fit


def accept_function(
    candidate,
    cur_fitness,
    cur_solution,
    best_fitness,
    best_solution,
    N,
    T,
    coords
):
    """
    Accept with probability 1 if candidate is better than current.
    Accept with probabilty p_accept(..) if candidate is worse.
    """
    candidate_fitness = fitness_function(candidate, N, coords)
    if candidate_fitness < cur_fitness:
        cur_fitness = candidate_fitness
        cur_solution = list(candidate)
        if candidate_fitness < best_fitness:
            best_fitness = candidate_fitness
            best_solution = list(candidate)
    else:
        if random.random() < p_accept_function(candidate_fitness, cur_fitness, T):
            cur_fitness = candidate_fitness
            cur_solution = list(candidate)

    return best_fitness, best_solution, cur_fitness, cur_solution


def anneal_cythran(
    cur_solution,
    cur_fitness,
    stopping_temperature,
    stopping_iter,
    T,
    N,
    iteration,
    fitness_list,
    best_fitness,
    best_solution,
    alpha,
    coords,
    candidate
):
    """
    Execute simulated annealing algorithm.
    """
    # Initialize with the greedy solution.
    # cur_solution, cur_fitness = initial_solution()

    print("Starting annealing.")
    while T >= stopping_temperature and iteration < stopping_iter:
        candidate = list(cur_solution)
        l = random.randint(2, N - 1)
        i = random.randint(0, N - l)
        last_index = i + l
        normal_index = i
        reversed_index = last_index - 1
        copied_candidate = candidate[:]

        while reversed_index >= i:
            candidate[reversed_index] = copied_candidate[normal_index]
            reversed_index -= 1
            normal_index += 1

        best_fitness, best_solution, cur_fitness, cur_solution = accept_function(
            candidate,
            cur_fitness,
            cur_solution,
            best_fitness,
            best_solution,
            N,
            T,
            coords
        )
        T *= alpha
        iteration += 1

        fitness_list.append(cur_fitness)
