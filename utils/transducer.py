import numpy as np
import matplotlib.pyplot as plt
from typing import *
from utils.agent import Agent
from utils.nets import Neuron
from collections import defaultdict


class Transducer:
    '''
    Takes agent responses and bounds in
    one-key hot coded distrubution

    A = P(W|R) for W in N
    R = P(AuB |A + B)
    '''

    def __init__(self, agents: list, question: str = "") -> None:
        '''
        Create bounded plane of responses for agent_a & agent_b
        '''
        # init
        self.agents = defaultdict()
        self.jaccard_indices = defaultdict()
        self.prob_vectors = defaultdict()
        self.r = defaultdict(int)
        self.all_responses: str = ""
        self.n: int
        self.neurons = defaultdict()
        # set values
        for agent in agents:
            if question != "":
                print('loading agent', agent.name, '... ')
                agent.chat_bot.one_question(question)

            self.agents[agent.name] = agent
            self.all_responses += agent.response

            for word in agent.response.split():
                self.r[word.lower()] += 1
        self.all_responses = set(self.all_responses.split())
        self.n = len(self.all_responses)
        self.bounded_space()

    def bounded_space(self) -> None:
        '''
        Calculates the Jaccard index based on the responses of two agents and 
        creates a one-hot encoded vector representing the word frequencies in both responses.

        Returns:
            tuple: A tuple containing the Jaccard index and a dictionary representing 
                the one-hot encoded vector of word frequencies.
        '''
        # Splitting responses into words and creating sets

        for name, agent in self.agents.items():
            response = set(agent.response.split())
            print('response', response)
            # Calculating the union and intersection of word sets
            union_words = response.union(self.all_responses)
            intersection_words = response.intersection(
                self.all_responses)

            # Calculating the Jaccard index
            jaccard_index = len(intersection_words) / \
                len(union_words) if union_words else 0

            self.jaccard_indices[name] = jaccard_index

            # Counting occurrences of each word in the combined responses
            self.prob_vectors[name] = self.prob_vect(response)

    def prob_vect(self, response: str) -> np.array:
        '''
        returns dict of probs of words
        '''
        prob_vect = [0] * len(self.r)
        word_freq = self.word_freq(response)
        idx = 0
        for key, _ in self.r.items():
            prob_vect[idx] = word_freq[key.lower()] / len(word_freq)
            idx += 1

        return prob_vect

        # for key, value in word_freq_dict:
        #     prob_vect[key] = value / self.n

    def word_freq(self, pasage: str) -> dict:
        '''
        returns a freq dict of words in passage
        '''
        res = defaultdict(int)

        for word in pasage:
            res[word.lower()] += 1

        return res

    def shannon_entropy(self, response: str) -> float:
        """
        Calculates the Shannon Entropy (H) of a dataset.

        Parameters:
        counts (List[float]): A list of probabilities.

        Returns:
        float: Shannon Entropy of the dataset.
        """
        counts = self.prob_vect(response)
        return -sum(p * np.log(p) if p > 0 else 0 for p in counts)

    def true_diversity(self, response: str) -> float:
        """
        Calculates the True Diversity (D) using Shannon Entropy (H).

        Parameters:
        counts (List[float]): A list of probabilities.

        Returns:
        float: True Diversity of the dataset.
        """
        counts = self.word_freq(response).values()
        return np.exp(self.shannon_entropy(counts))

    def create_layer(self) -> np.array:
        '''
        creates transducer layer for input
        '''

        main_vector = self.prob_vect(self.all_responses)
        idx = 0
        for name, agent in self.agents.items():
            prob_v = np.array(self.prob_vectors[name])
            delta_dot = np.dot(main_vector, prob_v)
            self.neurons[name] = Neuron(input=delta_dot, id=idx, layer=1)
            self.neurons[name].prov_v = prob_v
            self.neurons[name].dot_product = delta_dot
            self.neurons[name].cross = main_vector * prob_v
            self.neurons[name].jaccard_index = self.jaccard_indices[name]
            self.neurons[name].x = self.jaccard_indices[name]
            self.neurons[name].y = delta_dot
            self.neurons[name].z = self.shannon_entropy(agent.response)
            idx += 1


if __name__ == "__main__":
    embedding_params = ["facebook-dpr-ctx_encoder-multiset-base", 200, 25, 0.9]

    agent_ltoa = Agent(
        'agent_ltoa', 'chroma_db/agent_ltoa', 0, embedding_params, True)

    agent_norbert = Agent(
        'agent_norbert', 'chroma_db/agent_norbert', 0, embedding_params, True)

    input_layer = Transducer(
        [agent_ltoa, agent_norbert], "how can one design a neuron?")
