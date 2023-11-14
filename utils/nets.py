import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union, Dict, Tuple
from collections import defaultdict


class Neuron:
    """
    A class representing a neuron in a neural network.

    Attributes:
        input (float): The input value to the neuron.
        id (int): The identifier for the neuron.
        layer (int): The layer number in the network where the neuron is located.
        biases (float): The bias value for the neuron.
        neural_params (np.ndarray): The weight matrix of the neuron (3x2).
        state (float): The state of the neuron.
        signal (np.ndarray): The input signal to the neuron (3x1).
        edges (Dict): The edges connecting this neuron to others.
    """

    def __init__(self, input_value: float, neuron_id: int, layer: int = 0, mu: float = 1, sigma: float = 1, bias: float = 0.01):
        """
        Initializes the Neuron with given parameters and random state and weights.

        Args:
            input_value (float): The input value to the neuron.
            neuron_id (int): The identifier for the neuron.
            layer (int): The layer number in the network where the neuron is located.
            mu (float): The mean for the normal distribution used in state initialization.
            sigma (float): The standard deviation for the normal distribution used in state initialization.
            bias (float): The bias value for the neuron.
        """
        self.input: float = input_value
        self.id: int = neuron_id
        self.layer: int = layer
        self.biases: float = bias
        self.neural_params: np.ndarray = np.random.rand(3, 2)
        self.state: float = np.random.normal(mu, sigma)
        self.signal: np.ndarray = np.array([self.input, self.state, 1])
        self.edges: Dict = defaultdict()

    def activate(self, x: np.ndarray, use_sigmoid: bool = False) -> np.ndarray:
        """
        Applies the activation function.

        Args:
            x (np.ndarray): The input array to the activation function.
            use_sigmoid (bool): Flag to choose between tanh and sigmoid function.

        Returns:
            np.ndarray: The result of the activation function.
        """
        return 1 / (1 + np.exp(-x)) if use_sigmoid else np.tanh(x)

    def activate_derivative(self, x: np.ndarray, use_sigmoid: bool = False) -> np.ndarray:
        """
        Calculates the derivative of the activation function.

        Args:
            x (np.ndarray): The input array.
            use_sigmoid (bool): Flag to choose between tanh and sigmoid function derivative.

        Returns:
            np.ndarray: The derivative of the activation function.
        """
        return self.activate(x, use_sigmoid) * (1 - self.activate(x, use_sigmoid)) if use_sigmoid else 1.0 - np.tanh(x) ** 2

    def feed_forward(self) -> None:
        """
        Performs the feed forward operation of the neuron.
        Updates self.input & self.state inside self.signal.
        """
        z = np.dot(self.neural_params.T, self.signal)
        self.input, self.state = self.activate(z)
        self.signal = np.array([self.input, self.state, 1])

    def backprop(self, target: float, learning_rate: float) -> None:
        """
        Performs backpropagation for the neuron.

        Args:
            target (float): The target output for the neuron.
            learning_rate (float): The learning rate for weight updates.
        """
        error = (target - self.state) * self.activate_derivative(self.input)
        gradient = np.outer(self.signal, error)
        self.neural_params -= learning_rate * gradient
        self.biases -= learning_rate * error

    def train(self, X: np.ndarray, Y: np.ndarray, epochs: int = 100, learning_rate: float = 0.01) -> Tuple[List[np.ndarray], List[float], List[float]]:
        """
        Train the neuron on a dataset.

        Args:
            X (np.ndarray): Input features.
            Y (np.ndarray): Target output.
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for weight updates.

        Returns:
            Tuple[List[np.ndarray], List[float], List[float]]: A tuple containing the evolution of weights, biases, and errors during training.
        """
        weights_evolution = []
        biases_evolution = []
        errors_evolution = []

        for _ in range(epochs):
            epoch_errors = []
            for x, y in zip(X, Y):
                self.signal = np.array([x, self.state, 1])
                self.feed_forward()
                error = y - self.state
                epoch_errors.append(error ** 2)
                self.backprop(target=y, learning_rate=learning_rate)

            weights_evolution.append(self.neural_params.copy())
            biases_evolution.append(self.biases)
            errors_evolution.append(np.mean(epoch_errors))

        return weights_evolution, biases_evolution, errors_evolution


# Example usage
if __name__ == "__main__":
    # Create a Neuron instance
    neuron = Neuron(input_value=0, neuron_id=1)
    epochs = 10
    # Sample dataset (linear relationship for simplicity)
    X = np.array([0, 1, 2, 3, 4])
    Y = np.array([2, 4, 6, 8, 10])

    # Train the neuron
    weights, biases, errors = neuron.train(X, Y, epochs)

    # Plotting the results
    epochs = range(1, 11)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(epochs, [w[0][0] for w in weights], label='Weight 1')
    plt.plot(epochs, [w[0][1] for w in weights], label='Weight 2')
    plt.title('Weights Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, biases)
    plt.title('Biases Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Value')

    plt.subplot(1, 3, 3)
    plt.plot(epochs, errors)
    plt.title('Training Error')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')

    plt.tight_layout()
    plt.show()
