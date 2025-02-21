from itertools import compress
import random
import time
import matplotlib.pyplot as plt

from data import *


def initial_population(individual_size, population_size):
    return [[random.choice([True, False]) for _ in range(individual_size)] for _ in range(population_size)]


def fitness(items, knapsack_max_capacity, individual):
    total_weight = sum(compress(items['Weight'], individual))
    if total_weight > knapsack_max_capacity:
        return 0
    return sum(compress(items['Value'], individual))


def population_best(items, knapsack_max_capacity, population):
    best_individual = None
    best_individual_fitness = -1
    for individual in population:
        individual_fitness = fitness(items, knapsack_max_capacity, individual)
        if individual_fitness > best_individual_fitness:
            best_individual = individual
            best_individual_fitness = individual_fitness
    return best_individual, best_individual_fitness


def one_point_crossover(parent1, parent2):
    ind_size = len(parent1)
    child1 = parent1[:int(ind_size // 2)] + parent2[int((ind_size // 2)):]
    child2 = parent2[:int(ind_size // 2)] + parent1[int((ind_size // 2)):]
    return [child1, child2]


def random_parents(parents):
    p = random.sample(parents, 2)
    return p[0], p[1]


def mutate(individual):
    i = random.randint(0, len(individual) - 1)
    new_individual = individual.copy()
    new_individual[i] = not individual[i]
    return new_individual


def population_fitness(population, items, knapsack_max_capacity):
    pop_fitness = []
    for pop in population:
        pop_fitness.append(fitness(items, knapsack_max_capacity, pop))
    return pop_fitness


def calculate_roulette_wheel_probabilities(population_fitness):
    global_fitness = sum(population_fitness)
    probabilities = population_fitness.copy()
    for p in probabilities:
        p /= global_fitness
    return probabilities


def roulette_wheel_selection(population, probabilities, to_select):
    return random.choices(population=population, weights=probabilities, k=to_select)


items, knapsack_max_capacity = get_big()
print(items)

population_size = 100
generations = 200
n_selection = 20
n_elite = 10
children = 20
mutants = population_size - children - n_elite

start_time = time.time()
best_solution = None
best_fitness = 0
population_history = []
best_history = []
population = initial_population(len(items), population_size)

for gen in range(generations):
    population_history.append(population.copy())

    best_individual, best_individual_fitness = population_best(items, knapsack_max_capacity, population)
    if best_individual_fitness > best_fitness:
        best_solution = best_individual
        best_fitness = best_individual_fitness
    best_history.append(best_fitness)

    # calculate population fitness
    pop_fitness = population_fitness(population, items, knapsack_max_capacity)
    # take n_elite elite of most fit from the population
    elite = []
    for _ in range(n_elite):
        elitist_idx = pop_fitness.index(max(pop_fitness))
        elite.append(population.pop(elitist_idx))
        pop_fitness.pop(elitist_idx)
    # select n_selection - n_elite from population without the elite
    parents = roulette_wheel_selection(population,
                                       calculate_roulette_wheel_probabilities(pop_fitness),
                                       n_selection - n_elite)


    # crossover
    new_population = elite
    for _ in range(children):
        new_population += one_point_crossover(*random_parents(parents))

    # mutating
    for _ in range(mutants):
        new_population.append(mutate(random.choice(new_population)))

    # add elites into new population
    new_population = new_population + elite

    # update
    population = new_population
    print(f"Generation - {gen}: best fitness {best_individual_fitness}, pop fitness {sum(pop_fitness)}")


end_time = time.time()
total_time = end_time - start_time
print('Best solution:', list(compress(items['Name'], best_solution)))
print('Best solution value:', best_fitness)
print('Time: ', total_time)

# plot generations
x = []
y = []
top_best = 20
plt.figure(figsize=(20, 12))
for i, population in enumerate(population_history):
    plotted_individuals = min(len(population), top_best)
    x.extend([i] * plotted_individuals)
    population_fitnesses = [fitness(items, knapsack_max_capacity, individual) for individual in population]
    population_fitnesses.sort(reverse=True)
    y.extend(population_fitnesses[:plotted_individuals])
plt.scatter(x, y, marker='.')
plt.plot(best_history, 'r')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()
