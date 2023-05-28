import pandas as pd

data = pd.read_csv('data/train.csv')
output_data = data['label'].tolist()

def one_hot_encode(numbers):
    encoded_list = []
    for num in numbers:
        encoded_num = [0] * 10  # Create a list of 10 zeros
        encoded_num[num] = 1   # Set the index corresponding to the number to 1
        encoded_list.append(encoded_num)
    return encoded_list

def count_weights(neurons_per_layer):
    total_weights = 0
    for i in range(1, len(neurons_per_layer)):
        current_layer_neurons = neurons_per_layer[i]
        previous_layer_neurons = neurons_per_layer[i-1]
        weights_in_layer = current_layer_neurons * previous_layer_neurons
        total_weights += weights_in_layer
    return total_weights


Size_data_set = 300
LABEL_DATA = one_hot_encode(output_data)[:Size_data_set] #1 = [0,1,0,0,0,0,0,0,0,0]
INPUT_DATA = data.drop(columns='label')[:Size_data_set].values.tolist()

del data
del output_data

N_CLASSES = 10
SIZE_HIDDEN_LAYER = 10
size_input_layer = len(INPUT_DATA[0])
NUMBER_OF_WEIGHTS = count_weights([size_input_layer,SIZE_HIDDEN_LAYER,N_CLASSES])





