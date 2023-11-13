import numpy as np
import typing
import matplotlib.pyplot as plt
import networkx as nx
import heapq


class Neuron:

    '''
    weights: 3x2 weight matrix
    state: 1x3 state input
    Output: delta input
    '''

    def __init__(self, input: float, id: int, layer: int = 0, mu: float = 1, sigma: float = 1,  bias: float = 0.01):
        '''
        Creates a random neuron
        self.weights = 3x2
        self.signal = 3x1
        '''
        self.id = id
        self.layer = layer
        self.edges = []
        self.input = input
        self.bias = bias
        self.state = np.random.randint(-1, 1)
        # 3x2
        self.weights = np.random.rand(3, 2)
        # 1x3
        self.signal = np.array(
            [input, np.random.normal(mu, sigma), 1])

    # Training
    def feed_forward(self, signal):
        '''
        Feedforward in neural net uses the tanh activation function
        to bound answer within -1-1 range, introduces non-linearity
        '''
        # print('w', self.weights)

        self.weights = self.activate(self.weights.T * signal, False)
        # 2x3
        output = np.dot(self.weights, signal)
        # self.weights
        self.weights = self.weights.T
        # print(f'weights {self.weights}')
        self.input = output[0]
        self.state = output[1]
        # 3x1
        self.signal = np.array([self.input, self.state, 1])
        # recurrent (system dynamics by feeding input -> output -> input)
        # 2x3
       # self.weights = np.array(self.weights)
        return self

    def edges_delta(self, k: int = 5, layer: list = None, threshold: float = 0.5) -> heapq:
        '''
        finds knn closet edges :)
        i.e, does it fire or nah
        '''
        if not layer:
            return ValueError('not great, need a list')

        res = [(abs(neuron.input - self.input), neuron)
               for neuron in layer.neurons if abs(neuron.input - self.input) <= threshold]

        return res

    def backprop(self):
        '''
        TODO: Create :)
        '''
        pass

    # Activation Functions

    def activate(self, x: np.ndarray, sig: bool = False) -> np.ndarray:
        """The activation function (tanh is default)."""
        if sig:
            return 1 + 1/np.exp(-x)
        return np.tanh(x)

    def activate_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of the activation function."""
        return 1.0 - np.tanh(x) ** 2

    # Getters & Setters
    def get_state(self) -> list:
        '''
        show and return current state
        '''
        return self.state

    def set_state(self, vector: list):
        '''
        update state
        '''
        self.state = vector

    def get_weights(self) -> list:
        '''
        print & show weights
        '''
        return self.weights

    def set_weights(self, vector: list):
        '''
        set weights to vector
        '''
        self.weights = vector

    def set_id(self, id: int):
        '''
        set id for neural network
        '''
        self.id = id


if __name__ == "__main__":

    n1 = Neuron(1.089, 1)
    n2 = Neuron(n1.input, 2)
    n1.feed_forward(n1.signal)

    n2.feed_forward(n1.signal)
    n1.feed_forward(n2.signal)

    plt.plot(np.tanh(n1.signal), label='tahn')
    plt.plot(1 / (1 + np.exp(-n1.signal)), label='sigmoid')
    plt.legend()
    plt.show()
