import numpy as np
import matplotlib.pyplot as plt
from typing import *
import logging


class Knn:
    """
    A class to perform KNN search and visualize the results.

    Attributes:
        objects (List[object]): A list of custom objects for KNN search.
    """

    def __init__(self, objects: List[object]) -> None:
        self.objects = objects

    def search(self, query_point: np.ndarray, k: int = 0) -> List[object]:
        """
        Perform a K-Nearest Neighbors search.

        Args:
            query_point (np.ndarray): The 2D point to query for its nearest neighbors.
            k (int): The number of nearest neighbors to find. Default 0 will search based on threshold

        Returns:
            List[object]: The k-nearest neighbors.
        """
        # an agents vector magnitude must be shorter than 50% of connections by default
        threshold = np.mean([np.linalg.norm(agent.state)
                            for agent in self.objects])

        distances = [(obj, np.tanh(np.linalg.norm(np.array(obj.state) - np.array(query_point))))
                     for obj in self.objects]
        sorted_indices = np.argsort([item[1] for item in distances])
        if 1 > k:
            # np.linalg.norm() Turns vector into scalar
            # TODO create probabilities distro of edges based on network science of the human brain
            return [self.objects[i] for i in sorted_indices if np.linalg.norm(self.objects[i].state) < threshold and np.random.rand() >= 0.3]
        # KNN Path
        return [self.objects[i] for i in sorted_indices[:k]]

    def visualize(self, query_point: np.ndarray, k: int) -> None:
        """
        Visualize the results of KNN search.

        Args:
            query_point (np.ndarray): The 2D point to query for its nearest neighbors.
            k (int): The number of nearest neighbors to visualize.
        """
        nearest_neighbors = self.search(query_point, k)
        print('nearest neighbors')
        print(nearest_neighbors)
        plt.figure()
        x_vals, y_vals = zip(*[obj.state for obj in self.objects])
        plt.scatter(x_vals, y_vals, label='Agents')

        # Highlight the K nearest neighbors
        knn_x, knn_y = zip(*[obj.state for obj in nearest_neighbors])
        plt.scatter(knn_x, knn_y, color='r', label='K Nearest Neighbors')

        # Query point
        plt.scatter(*query_point, color='g', label='Query Point')

        plt.legend()
        plt.show()


# if __name__ == "__main__":
#     embedding_params = ["facebook-dpr-ctx_encoder-multiset-base", 200, 25, 0.7]
#     entity_snd = Agent('agent_snd', 'chroma_db/agent_snd',
#                        embedding_params, 0, True)
#     entity_snd.knn.search([0.178, -0.217])
