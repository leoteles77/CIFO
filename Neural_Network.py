import numpy as np
from scipy.special import softmax
import math

import numpy as np

class SingleLayerNeuralNetwork:
    def __init__(self, weights, inputs, n_classes, size_hidden_layer):
        weights_first_layer = []
        weights_second_layer = []
        i=0
        for _ in range(len(inputs)):
            weights_first_layer.append(weights[i:i+size_hidden_layer])
            i+=size_hidden_layer
        
        for _ in range(size_hidden_layer):
            weights_second_layer.append(weights[i:i+n_classes])
            i+=n_classes

        self.weights_first_layer = np.array(weights_first_layer).transpose()
        self.weights_second_layer = np.array(weights_second_layer).transpose()
        self.input = np.array(inputs)
    
    def relu(self,x):
        return(np.maximum(0, x))
    
    def foward(self):
        values_first_layer = self.weights_first_layer.dot(self.input)
        #activation function RELU
        values_first_layer = self.relu(values_first_layer)

        #SECOND LAYER
        values_second_layer = self.weights_second_layer.dot(values_first_layer)
        values_second_layer = softmax(values_second_layer)

        return values_second_layer