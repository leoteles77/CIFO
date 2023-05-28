from random import shuffle, choice, sample, random
from operator import attrgetter
from copy import deepcopy
import numpy as np
import pandas as pd
import time
from data.data import INPUT_DATA, LABEL_DATA,N_CLASSES,SIZE_HIDDEN_LAYER
from Neural_Network import SingleLayerNeuralNetwork
import sys

class Individual:
    def __init__(self,representation=None,size=None):
        if representation == None and size != None:
            self.representation = np.random.normal(loc=0, scale=1, size=size) * 0.3
        else:
            self.representation = representation
        self.fitness = self.get_fitness()

    def get_fitness(self):
        raise Exception("You need to monkey patch the fitness path.")

    def get_representation(self):
        return self.representation
        
    def index(self, value):
        return self.representation.index(value)

    def __len__(self):
        return len(self.representation)

    def __getitem__(self, position):
        return self.representation[position]

    def __setitem__(self, position, value):
        self.representation[position] = value

    def __repr__(self):
        return f"Individual(Fitness: {self.fitness})"

class Population:
    def __init__(self, size, optim, **kwargs):
        self.individuals = []
        self.size = size
        self.optim = optim
        for _ in range(size):
            self.individuals.append(
                Individual(
                    representation=kwargs["representation"],
                    size=kwargs["number_of_weights"]
                )
            )

    def evolve(self, iteration, file_name, gens, select, crossover, mutate, xo_p, mut_p, elitism=True, n_elitism=1):
        column_names = ['iteration', 'gen', 'bestfitness','time']
        df = pd.DataFrame(columns=column_names)
        for gen in range(gens):
            start_time = time.time()
            new_pop = []

            if elitism:
                if self.optim == 'max':
                    elite_list = deepcopy(sorted(self.individuals, key=attrgetter("fitness"), reverse=True)[:n_elitism])
                elif self.optim == 'min':
                    elite_list = deepcopy(sorted(self.individuals, key=attrgetter("fitness"), reverse=False)[:n_elitism])

            while len(new_pop) < self.size:
                #CROSSOVER
                parent1, parent2 = select(self), select(self)
                if random() < xo_p:
                    offspring1, offspring2 = crossover(parent1, parent2)
                else:
                    offspring1, offspring2 = parent1, parent2

                #MUTATION
                if random() < mut_p:
                    offspring1 = mutate(offspring1)
                if random() < mut_p:
                    offspring2 = mutate(offspring2)

                new_pop.append(Individual(representation=offspring1))
                if len(new_pop) < self.size:
                    new_pop.append(Individual(representation=offspring2))

            if elitism:
                if self.optim == "max":
                    worse_list = sorted(new_pop, key=attrgetter("fitness"), reverse=False)[:n_elitism]
                if self.optim == "min":
                    worse_list = sorted(new_pop, key=attrgetter("fitness"), reverse=True)[:n_elitism]
                
                for i in worse_list:
                    new_pop.pop(new_pop.index(i))
                for i in elite_list:
                    new_pop.append(i)


            self.individuals = new_pop

            if self.optim == "max":
                print(gen, f' : Best individual: {max(self, key=attrgetter("fitness"))}')
                best_ind = max(self, key=attrgetter("fitness")).fitness
                best_representation = max(self, key=attrgetter("fitness")).representation
            elif self.optim == "min":
                print(gen, f' : Best individual: {min(self, key=attrgetter("fitness"))}')
                best_ind = min(self, key=attrgetter("fitness")).fitness
                best_representation = min(self, key=attrgetter("fitness")).representation
            
            total_time = time.time() - start_time

            new_row = {'iteration':iteration+1, 'gen':gen + 1, 'bestfitness':best_ind, 'time':total_time}
            df = df._append(new_row, ignore_index=True)

            def calculate_accuracy(predictions, labels):
                # Convert predictions to class labels by selecting the index of the maximum value
                predicted_labels = np.argmax(predictions, axis=1)
                labels_true = np.argmax(labels, axis=1) 
                # Compare predicted labels with ground truth labels
                correct_predictions = np.equal(predicted_labels, labels_true)
                # Calculate accuracy as the percentage of correct predictions
                accuracy = np.mean(correct_predictions)

                return accuracy

            prob = []
            for i in INPUT_DATA:
                net_eval = SingleLayerNeuralNetwork(weights=best_representation,inputs=i,n_classes=N_CLASSES,size_hidden_layer=SIZE_HIDDEN_LAYER)
                prob.append(net_eval.foward())
            print('Accuracy:',calculate_accuracy(prob,LABEL_DATA))


        if iteration == 0:
            df.to_csv('results/' + file_name+'.csv', mode='a', index=False, header=column_names)
        else:
            df.to_csv('results/' + file_name+'.csv', mode='a', index=False, header=False)

    def __len__(self):
        return len(self.individuals)

    def __getitem__(self, position):
        return self.individuals[position]