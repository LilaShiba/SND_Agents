from typing import List, Any, Tuple
import numpy as np
from collections import Counter
import logging
from scipy.stats import wasserstein_distance
import time


class ThoughtDiversity:
    """
    Class to measure the diversity of thought in an AI agent composed of several agents.
    """

    def __init__(self, pack: Any) -> None:
        """
        Initializes the ThoughtDiversity instance with an Agent_Pack instance.

        Parameters:
        pack (Any): An instance of Agent_Pack.
        """
        self.pack = pack
        self.scores = []
        self.snd_scores = []
        self.jaccard_indexes = []
        self.current_mcs_samples = []
        self.shannon_entropy_scores = []
        self.true_diversity_scores = []
        self.wasserstein_metrics = []
        self.ginis = []

    def monte_carlo_sim(self, question: str = "", rounds: int = 5) -> List[Any]:
        """
        Run a Monte Carlo simulation to assess thought diversity.

        Parameters:
        question (str): The question to be asked to the agent pack.
        rounds (int): The number of rounds to run the simulation.

        Returns:
        List[Any]: The results of the Monte Carlo simulation, including Shannon Entropy scores,
                   True Diversity scores, and Wasserstein metrics.
        """
        results = []

        for _ in range(rounds):
            round_res = self.pack.one_question(question)
            time.sleep(60)
            # Extract responses from individual agents
            responses = [round_res[agent_name]
                         for agent_name in self.pack.agent_names if agent_name in round_res]

            if responses:
                # Processing each agent's response
                for response in responses:
                    prob_vector = self._prob_vectors(response)
                    entropy = self.shannon_entropy(prob_vector)
                    diversity = self.true_diversity(prob_vector)
                    self.shannon_entropy_scores.append(entropy)
                    self.true_diversity_scores.append(diversity)

                # Calculating Wasserstein metrics
                for i in range(len(responses)):
                    for j in range(i + 1, len(responses)):
                        w_metric = self.wasserstein(self._prob_vectors(
                            responses[i]), self._prob_vectors(responses[j]))
                        self.wasserstein_metrics.append(w_metric)

        return [self.shannon_entropy_scores, self.true_diversity_scores, self.wasserstein_metrics]

    def _prob_vectors(self, vector: List[str]) -> List[float]:
        """
        Create probability vectors from a list of responses.

        Parameters:
        vector (List[str]): A list of string responses.

        Returns:
        List[float]: A list representing the probability vector.
        """
        joined_strings = ' '.join(vector)
        words = joined_strings.split()
        word_counts = Counter(words)
        total_counts = sum(word_counts.values())
        return [count / total_counts for count in word_counts.values()]

    def shannon_entropy(self, counts: List[float]) -> float:
        """
        Calculates the Shannon Entropy (H) of a dataset.

        Parameters:
        counts (List[float]): A list of probabilities.

        Returns:
        float: Shannon Entropy of the dataset.
        """
        return -sum(p * np.log(p) if p > 0 else 0 for p in counts)

    def true_diversity(self, counts: List[float]) -> float:
        """
        Calculates the True Diversity (D) using Shannon Entropy (H).

        Parameters:
        counts (List[float]): A list of probabilities.

        Returns:
        float: True Diversity of the dataset.
        """
        return np.exp(self.shannon_entropy(counts))

    def wasserstein(self, p1: List[float], p2: List[float]) -> float:
        """
        Calculates the Wasserstein distance between two probability vectors.

        Parameters:
        p1, p2 (List[float]): Probability vectors.

        Returns:
        float: Wasserstein distance between p1 and p2.
        """
        return wasserstein_distance(p1, p2)
