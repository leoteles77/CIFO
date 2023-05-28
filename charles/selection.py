from random import uniform, sample, randint
from operator import attrgetter
import numpy as np

def fps(population):
    if population.optim == "max":
        # Computes the totallity of the population fitness
        population_fitness = sum([chromosome.fitness for chromosome in population])
        # Computes for each chromosome the probability 
        chromosome_probabilities = [chromosome.fitness/population_fitness for chromosome in population]
        # Selects one chromosome based on the computed probabilities
        return population[np.random.choice(len(population), p=chromosome_probabilities)]

    elif population.optim == "min":
        # Computes the totallity of the population fitness
        population_fitness = sum([chromosome.fitness for chromosome in population])
        # Computes for each chromosome the probability 
        chromosome_probabilities = [chromosome.fitness/population_fitness for chromosome in population]
        # Making the probabilities for a minimization problem
        chromosome_probabilities2 = 1 / np.array(chromosome_probabilities)
        #Normalize
        sum_normal = sum(chromosome_probabilities2)
        new_probabilities = chromosome_probabilities2/sum_normal
        # Selects one chromosome based on the computed probabilities
        return population[np.random.choice(len(population), p=new_probabilities)]
    else:
        raise Exception("No optimization specified (min or max).")

def tournament_sel(population, size=4):
    tournament = sample(population.individuals, size)
    if population.optim == "max":
        return max(tournament, key=attrgetter("fitness"))
    if population.optim == "min":
        return min(tournament, key=attrgetter("fitness"))

def rank (population):
    if population.optim == "max":
        population.individuals.sort(key=attrgetter("fitness"))
    elif population.optim == "min":
        population.individuals.sort(key=attrgetter("fitness"), reverse=True)
    else:
        raise Exception("No optimization specified (min or max).")
    
    total = sum(range(population.size+1))
    chromosome_probabilities = [i/total for i in range(1, len(population) + 1)]
    return population[np.random.choice(len(population), p=chromosome_probabilities)]