# First attempt at problem, using only cost function for training
# Heavy inspiration from 3blue1brown video

import numpy as np

# Reading data from file, as csv form

test_data_raw = open("mnist_test.csv", "r")

# First index of each list is numerical value, rest is pixel data
test_data = [x.split(",") for x in test_data_raw.read().split("\n")] 

# Creating the neural network

## Scaled sigmoid function
def sigmoid(x) :
    return 2*((1 / (1 + np.exp(-x))-0.5))

## Define the sizes of the network layers

input_size = 784
hidden1_size = 16
hidden2_size = 16
output_size = 10

test_number = test_data[1][1:] # should be 7
test_input = np.asarray([sigmoid(int(x)) for x in test_number])

## Initialise with random Xavier weights and 0 biases for each layer

input_to_hidden1_weights = np.random.rand(input_size, hidden1_size) * np.sqrt(2 / (input_size + hidden1_size))
input_to_hidden1_biases = np.zeros(hidden1_size)

hidden1_to_hidden2_weights = np.random.rand(hidden1_size, hidden2_size) * np.sqrt(2 / (hidden1_size + hidden2_size))
hidden1_to_hidden2_biases = np.zeros(hidden2_size)

hidden2_to_output_weights = np.random.rand(hidden2_size, output_size) * np.sqrt(2 / (hidden2_size + output_size))
hidden2_to_output_biases = np.zeros(output_size)



## Function to run the network on an input

def passForward(input_neurons, weights, biases):
    return np.vectorize(sigmoid)(np.add(np.matmul(weights.transpose(), input_neurons), biases))


def think(input):
    a = passForward(input, input_to_hidden1_weights, input_to_hidden1_biases)
    b = passForward(a, hidden1_to_hidden2_weights, hidden1_to_hidden2_biases)
    return passForward(b, hidden2_to_output_weights, hidden2_to_output_biases)

print(think(test_input))
print(np.argmax(think(test_input)))
