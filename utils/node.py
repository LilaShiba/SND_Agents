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
            self.input: float = np.random.power(-1, 1)
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


if __name__ == "__main__":
    X = 0.4444
    y = 0.3333
    n1 = Neuron('n1', 1, X, 3)
    n2 = Neuron('n2', 2, X, 3)
    print(n1.feedforward(n2))
