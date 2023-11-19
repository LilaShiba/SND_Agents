import numpy as np
from typing import *
import matplotlib.pyplot as plt
from collections import defaultdict


class Neuron:

    '''
    represent an agent as a node
    in a neural network
    '''

    def __init__(self, id: str, layer: int, input: float = None, vector_size: int = 3) -> object:
        '''
        node class to represent diverse connections
        As an act a lone OR
        a wrapper for agent classes
        '''
        # weights
        self.weights: np.array = np.random.rand(2, 3)
        self.vector: np.array = np.random.rand(1, 3)
        self.vector[-1] = 1
        # input power law distro (to mimic actual neurons)
        if not input:
            self.input: float = np.random.normal(np.pi) * 2
        else:
            self.input: float = input
        # Direction
        self.heading: float = np.random.normal(np.pi) * 2
        # Attrubuites
        self.layer: int = layer
        self.biases: float = np.random.power(1)
        self.id: str = id
        # activation function
        self.sig: bool = False
        self.vector = np.random.rand(vector_size)

    def feedforward(self, node: object) -> object:
        '''
        propagates signal through neuron
        '''
        # delta = [self.vector] * node.weights.T
        delta = np.dot(self.weights, self.vector.T)
        delta_input, delta_state = self.activate(delta)
        self.vector = np.array([delta_input, delta_state, 1])
        return delta_input, delta_state

    def activate(self, x: np.ndarray):
        '''
        process a neurons signal
        '''
        if self.sig:
            return 1/(1+np.exp(-x))
        return np.tanh(x)

    def activation_derivative(self, x: np.array):
        '''
        Derivative of the activation function
        '''
        return 1.0 - np.tanh(x) ** 2

    def backpropagate(self, target: float, learning_rate: float) -> float:
        """
        Compute the error and update weights and biases
        """
        # Compute error (difference between target and actual output)
        error = target - self.vector[0]

        # Compute gradients for weights and biases
        d_weights = error * \
            self.activation_derivative(self.vector[0]) * self.vector
        d_biases = error * self.activation_derivative(self.vector[0])

        # Update weights and biases
        self.weights += learning_rate * d_weights.T
        self.biases += learning_rate * d_biases

        return error ** 2  # Return squared error


class Network:

    def __init__(self, neurons: list) -> object:
        '''
        creates and trains network of neurons
        neurons: list (tuple (layers, neurons))
        '''

        self.layers = defaultdict()

        for idx, delta_layer in enumerate(neurons):
            self.layers[idx] = [Neuron(str(x), idx)
                                for x in range(delta_layer)]


# Example usage
if __name__ == "__main__":
    neuron = Neuron('n1', 1, 0.5, 3)
    inputs = [0.1, 0.2, 0.3, 0.4, 0.5]
    targets = [0.2, 0.4, 0.6, 0.4, 2.0]  # Example target values
    nn = Network([64, 128, 64])
    print(nn.layers)
    # errors = neuron.train(inputs, targets, learning_rate=0.01, epochs=50)
    # neuron.graph(errors)
