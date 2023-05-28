from joblib import Parallel, delayed
from charles.charles import Population, Individual
import numpy as np
import pandas as pd
from Neural_Network import SingleLayerNeuralNetwork
from data.data import INPUT_DATA, NUMBER_OF_WEIGHTS, LABEL_DATA, N_CLASSES, SIZE_HIDDEN_LAYER
from charles.crossover import single_point_co,arithmetic_xo,blend_crossover, uniform_crossover, geometric_crossover
from charles.mutation import swap_mutation, inversion_mutation, scramble_mutation, geometric_semantic_mutation
from charles.selection import tournament_sel, rank, fps
from fitness_functions import cross_entropy_loss, brier_score
import time
import os

USE_BRIER = False

def create_dir(dir_name):
    isExist = os.path.exists(dir_name)
    if not isExist:
        os.mkdir(dir_name)
    

def networkEval(iteration, params):
    def get_fitness(self):
        probabilities_per_class = []
        for image in INPUT_DATA:
            Neural_Network = SingleLayerNeuralNetwork(weights=self.representation, inputs=image, n_classes=N_CLASSES, size_hidden_layer=SIZE_HIDDEN_LAYER)
            probabilities_per_class.append(Neural_Network.foward())
        if USE_BRIER:
            loss = brier_score(y_true = LABEL_DATA, y_pred=probabilities_per_class)
        else:
            loss = cross_entropy_loss(y_true=np.array(LABEL_DATA), y_pred=probabilities_per_class)
        return loss

    # Monkey patching
    Individual.get_fitness = get_fitness

    pop = Population(
        size=params['size'],
        number_of_weights=NUMBER_OF_WEIGHTS,
        representation=None,
        optim="min"
    )

    pop.evolve(
        iteration=iteration,
        file_name=params['file_name'],
        gens=params['gens'],
        select=params['select'],
        mutate=params['mutate'],
        crossover=params['crossover'],
        xo_p=params['xo_p'],
        mut_p=params['mut_p'],
        elitism=params['elitism'],
        n_elitism=params['n_elitism']
    )



parallel = Parallel(n_jobs=-1)
n_runs = range(16)

