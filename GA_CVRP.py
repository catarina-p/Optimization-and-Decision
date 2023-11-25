import random
import time

import numpy as np
import RegExService

fileName = "E-n101-k14.txt"
max_generations = 10000
populationSize = 30


def get_data():
    capacityLimit, graph, demand, optimalValue = RegExService.getData(fileName)
    vertices = list(graph.keys())
    vertices.remove(1)

    edges = {(min(a, b), max(a, b)): np.sqrt((graph[a][0] - graph[b][0]) ** 2 + (graph[a][1] - graph[b][1]) ** 2) for
             a in graph.keys() for b in graph.keys()}

    return vertices, edges, capacityLimit, demand, optimalValue


def generate_genome(vertices, demand, capacityLimit):
    full_size = len(vertices)
    city = np.zeros(full_size, dtype=int)
    genome = list()
    while len(vertices) != 0:
        city[0] = np.random.choice(vertices)
        capacity = capacityLimit - demand[city[0]]
        vertices.remove(city[0])
        a = 1
        while len(vertices) != 0:
            city[a] = np.random.choice(vertices)
            capacity = capacity - demand[city[a]]
            if capacity > 0:
                vertices.remove(city[a])
                a = a + 1
            else:
                route = city[0:a]
                genome.append(route)
                break
        if len(vertices) != 0:
            city = np.zeros(full_size, dtype=int)
        else:
            route = city[0:a]
            genome.append(route)
    return genome


def generate_population(populationSize, vertices, demand, capacityLimit):
    population = list()
    for _ in range(populationSize):
        pop = generate_genome(vertices.copy(), demand, capacityLimit)
        population.append(pop)
    return population


def fitness_func(genome, edges):
    fitness = 0
    for i in range(len(genome)):
        for k in range(genome[i].shape[0]):
            if k == 0:
                fitness = fitness + edges[(1, genome[i][k])]  # sai do depot
            else:
                fitness = fitness + edges[
                    (min(genome[i][k - 1], genome[i][k]), max(genome[i][k - 1], genome[i][k]))]  # rota
        fitness = fitness + edges[(1, genome[i][k])]  # Volta ao depot
    return fitness


def selection_pair(population, edges, optimalValue):
    fitness_vector = list()
    for i in range(populationSize):
        fitness = fitness_func(population[i], edges)
        fitness_vector.append(fitness)
    selection_weights = [optimalValue / (x - optimalValue) for x in fitness_vector]
    genome_a, genome_b = random.choices(population, weights=selection_weights, k=2)
    return genome_a, genome_b


def alternating_edges_crossover(genome_a, genome_b, vertices, demand, capacityLimit, first_parent):
    child = list()
    routes_start = list()
    parents = [genome_a, genome_b]
    full_size = len(vertices)
    new_child = np.zeros(full_size, dtype=int)
    routes_start_a = [genome_a[r][0] for r in range(len(genome_a))]  # 1os nos de todas as rotas de p1
    routes_start_b = [genome_b[r][0] for r in range(len(genome_b))]  # 1os nos de todas as rotas de p2
    routes_start.append(routes_start_a)
    routes_start.append(routes_start_b)

    a = first_parent  # indica o genoma parente em que se começa
    while len(vertices) != 0:
        if len(routes_start[a]) != 0:  # se houver nós, dos 1os em cada rota que ainda nao foram visitados,
            # esses tbm deverão ser os 1os nos da rota do filho
            while len(routes_start[a]) != 0:
                value = np.random.choice(routes_start[a])
                if value in vertices:
                    new_child[0] = value
                    vertices.remove(new_child[0])
                    routes_start[a].remove(new_child[0])
                    break
                else:
                    routes_start[a].remove(value)
        if (len(routes_start[a]) == 0) and (new_child[0] == 0):  # caso contrário escolhe-se um aleatório para ser o 1º
            if np.size(vertices, axis=None) != 0:
                new_child[0] = np.random.choice(vertices)
                vertices.remove(new_child[0])
            else:
                break

        capacity = capacityLimit - demand[new_child[0]]

        c = 1
        while len(vertices) != 0:
            for k in range(len(parents[a])):
                idx = np.where(parents[a][k] == new_child[c - 1])  # encontrar o arco correspondente
                if np.size(idx, axis=None) != 0:
                    if (idx[0] + 1 < len(parents[a][k])) and (parents[a][k][idx[0] + 1] in vertices):
                        new_child[c] = parents[a][k][idx[0] + 1]
                        a = abs(a - 1)
                        break

            if new_child[c] == 0:  # se o no encontrado estiver no fim de uma rota, o proximo no do filho é aleatório
                if np.size(vertices, axis=None) != 0:
                    new_child[c] = np.random.choice(vertices)
                else:
                    child.append(new_child[0:c])
                    new_child = np.zeros(full_size, dtype=int)
                    break

            capacity = capacity - demand[new_child[c]]
            if capacity <= 0:  # se a capacidade for excedida -> começar nova rota
                child.append(new_child[0:c])
                new_child = np.zeros(full_size, dtype=int)
                break
            if np.size(vertices, axis=None) != 0:
                vertices.remove(new_child[c])
            else:
                break

            c = c + 1
    child.append(new_child[0:c])
    return child


#  Mutação de apenas uma rota
def mutation_func(genome):
    p = random.randrange(3)
    # Swap Mutation
    if p == 0:
        route2swap = random.randrange(len(genome))
        entry1 = random.randrange(np.size(genome[route2swap], axis=None))
        entry2 = random.randrange(np.size(genome[route2swap], axis=None))
        mutation = genome[route2swap][entry1]
        genome[route2swap][entry1] = genome[route2swap][entry2]
        genome[route2swap][entry2] = mutation

    # Scramble Mutation
    elif p == 1:
        route = random.randrange(len(genome))
        route2scramble = list(genome[route])
        i = 0
        while len(route2scramble) != 0:
            genome[route][i] = np.random.choice(route2scramble)
            route2scramble.remove(genome[route][i])
            i = i + 1

    # Inverse Mutation
    elif p == 2:
        route = random.randrange(len(genome))
        entry1 = random.randrange(np.size(genome[route], axis=None))
        entry2 = random.randrange(np.size(genome[route], axis=None))
        route2invert = list(genome[route][min(entry1, entry2):max(entry1, entry2)])
        for k in range(len(route2invert)):
            genome[route][min(entry1, entry2) + k] = route2invert[len(route2invert) - k - 1]

    return genome


# Generational
def run_evolution():
    vertices, edges, capacityLimit, demand, optimalValue = get_data()
    population = generate_population(populationSize, vertices.copy(), demand, capacityLimit)

    for g in range(max_generations + 1):
        population = sorted(population, key=lambda genome: fitness_func(genome, edges), reverse=False)

        next_generation = population[0:2]

        for j in range(int(len(population) / 2) - 1):
            genome_a, genome_b = selection_pair(population, edges, optimalValue)
            child_a = alternating_edges_crossover(genome_a, genome_b, vertices.copy(), demand, capacityLimit, 0)
            child_b = alternating_edges_crossover(genome_a, genome_b, vertices.copy(), demand, capacityLimit, 1)
            mutant_a = mutation_func(child_a)
            mutant_b = mutation_func(child_b)
            next_generation += [mutant_a, mutant_b]

        population = next_generation

    population = sorted(population, key=lambda genome: fitness_func(genome, edges), reverse=False)
    best_fitness = fitness_func(population[0], edges)

    return population, g, best_fitness


start = time.time()
population, generations, best_fitness = run_evolution()
end = time.time()
print(f"time: {end - start}s")
print(f"Number of generations: " + str(generations))
print("Solution: " + str(population[0]))
print(f"Fitness: " + str(best_fitness))
