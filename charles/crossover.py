from random import randint, sample, uniform, random
import numpy as np

def single_point_co(p1, p2):
    co_point = randint(1, len(p1)-2)
    
    offspring1 = np.concatenate((p1[:co_point], p2[co_point:]))
    offspring2 = np.concatenate((p2[:co_point], p1[co_point:]))

    return list(offspring1), list(offspring2)

def arithmetic_xo(p1,p2):
    offspring1 = [None]*len(p1)
    offspring2 = [None]*len(p2)

    #set a value for alfa between 0 and 1
    alpha = uniform(0,1)

    # take the wieghts sum of the parents according to the formula
    for i in range(len(p1)):
        offspring1[i] = p1[i] * alpha + (1-alpha) * p2[i]
        offspring2[i] = p2[i] * alpha + (1-alpha) * p1[i]

    return offspring1, offspring2

def blend_crossover(parent1, parent2, alpha=1):
    offspring1 = np.zeros_like(parent1)
    offspring2 = np.zeros_like(parent2)

    for i in range(len(parent1)):
        diff = abs(parent1[i] - parent2[i])
        lower_bound = min(parent1[i], parent2[i]) - alpha * diff
        upper_bound = max(parent1[i], parent2[i]) + alpha * diff

        offspring1[i] = np.random.uniform(lower_bound, upper_bound)
        offspring2[i] = np.random.uniform(lower_bound, upper_bound)

    return list(offspring1), list(offspring2)

def uniform_crossover(parent1, parent2, crossover_prob=0.5):
    offspring1 = np.zeros_like(parent1)
    offspring2 = np.zeros_like(parent2)

    for i in range(len(parent1)):
        if np.random.rand() <= crossover_prob:
            offspring1[i] = parent2[i]
            offspring2[i] = parent1[i]
        else:
            offspring1[i] = parent1[i]
            offspring2[i] = parent2[i]

    return list(offspring1), list(offspring2)

def geometric_crossover(parent1, parent2):
    def geometric_xo(parent1,parent2):
        offspring = []
        for i in range(len(parent1)):
            ri = uniform(0, 1)
            offspring.append(ri * parent1[i] + (1 - ri) * parent2[i])
        return offspring
    offspring1, offspring2 = geometric_xo(parent1,parent2), geometric_xo(parent1,parent2)
    return offspring1, offspring2