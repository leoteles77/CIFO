from random import randint
from random import sample, uniform
import numpy as np

def swap_mutation(individual):
    mut_indexes = sample(range(len(individual)),2)
    individual[mut_indexes[0]], individual[mut_indexes[1]] = individual[mut_indexes[1]], individual[mut_indexes[0]]
    return individual

def inversion_mutation(individual):
    mut_points = sample(range(0,len(individual)),2)
    mut_points.sort()
    individual[mut_points[0]:mut_points[1]] = individual[mut_points[0]:mut_points[1]][::-1]
    return individual

def scramble_mutation(individual):
    mut_points = sample(range(0,len(individual)),2)
    mut_points.sort()
    teste = np.array(individual[mut_points[0]:mut_points[1]])
    np.random.shuffle(teste)
    individual[mut_points[0]:mut_points[1]] = teste
    return individual

def geometric_semantic_mutation(individual, mutation_step=0.5):
    mutated_individual = []
    for i in range(len(individual)):
        ri = uniform(-mutation_step, mutation_step)
        mutated_value = individual[i] + ri
        mutated_individual.append(mutated_value)
    return mutated_individual