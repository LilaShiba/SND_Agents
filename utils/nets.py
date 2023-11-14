from collections import defaultdict
import heapq
from typing import *

import numpy as np


class Neuron:
    """A class representing a neuron in a neural network.

    Attributes:
        input (float): The input value to the neuron.
        id (int): The identifier for the neuron.
        layer (int): The layer number in the network where the neuron is located.
        mu (float): The mean for the normal distribution used in state initialization.
        sigma (float): The standard deviation for the normal distribution used in state initialization.
        biases (float): The bias value for the neuron.
        neural_params (np.ndarray): The weight matrix of the neuron (3x2).
        state (float): The state of the neuron.
        signal (np.ndarray): The input signal to the neuron (3x1).
        edges (Dict): The edges connecting this neuron to others.
    """

    def __init__(self,
                 input: float,
                 id: int,
                 layer: int = 0,
                 mu: float = 1,
                 sigma: float = 1,
                 bias: float = 0.01):
        """Initializes the Neuron with given parameters and random state and weights."""
        self.input: float = input
        self.id: int = id
        self.layer: int = layer
        self.biases: float = bias
        self.neural_params: np.ndarray = np.random.rand(3, 2)
        self.state: float = np.random.normal(mu, sigma)
        self.signal: np.ndarray = np.array(
            [self.input, self.state, 1])
        self.edges: Dict = defaultdict()

    def activate(self, x: np.ndarray, sig: bool = False) -> np.ndarray:
        """Applies the activation function.

        Args:
            x (np.ndarray): The input array to the activation function.
            sig (bool): Flag to choose between tanh and sigmoid function.

        Returns:
            np.ndarray: The result of the activation function.
        """
        if sig:
            return 1 / (1 + np.exp(-x))
        return np.tanh(x)

    def activate_derivative(self, x: np.ndarray) -> np.ndarray:
        """Calculates the derivative of the activation function.

        Args:
            x (np.ndarray): The input array.

        Returns:
            np.ndarray: The derivative of the activation function.
        """
        return 1.0 - np.tanh(x) ** 2

    def feed_forward(self, signal: np.array = None) -> None:
        """
        Performs the feed forward operation of the neuron.
        Updates self.input & self.state inside self.signal

        """
        z: np.ndarray = np.dot(self.neural_params.T, self.signal)
        self.input, self.state = self.activate(z, False)
        self.signal = [self.input, self.state, 1]
        print(self.signal)

    def edges_delta(self, k: int = 5, layer: List['Neuron'] = None, threshold: float = 0.5) -> List[Union[float, 'Neuron']]:
        """Finds k-nearest neighboring edges based on the input difference.

        Args:
            k (int): Number of nearest neighbors to find.
            layer (List[Neuron]): The layer of neurons to search within.
            threshold (float): The threshold for considering a neuron as a neighbor.

        Returns:
            List[Union[float, Neuron]]: A list of tuples containing the difference in input and the neighboring neuron.
        """
        if not layer:
            raise ValueError('Layer is required and cannot be None.')

        res: List[Union[float, 'Neuron']] = [(abs(neuron.input - self.input), neuron)
                                             for neuron in layer if abs(neuron.input - self.input) <= threshold]
        return heapq.nsmallest(k, res, key=lambda x: x[0])

    def backprop(self) -> None:
        """Backpropagation method (to be implemented)."""
        pass

    # Getters & Setters
    def get_state(self) -> float:
        """Returns the current state of the neuron."""
        return self.state

    def set_state(self, state: float) -> None:
        """Updates the state of the neuron.

        Args:
            state (float): The new state value.
        """
        self.state = state

    def get_weights(self) -> np.ndarray:
        """Returns the weights of the neuron."""
        return self.neural_params

    def set_weights(self, weights: np.ndarray) -> None:
        """Sets the weights of the neuron.

        Args:
            weights (np.ndarray): The new weight matrix.
        """
        self.neural_params = weights

    def set_id(self, id: int) -> None:
        """Sets the ID of the neuron.

        Args:
            id (int): The new ID value.
        """
        self.id = id


if __name__ == "__main__":
    neuron = Neuron(0.5, 1)
    neuron.feed_forward()
