import numpy as np
import matplotlib.pyplot as plt
from typing import *
from utils.agent import Agent
from collections import defaultdict


class Transducer:
    '''
    Takes agent responses and bounds in
    one-key hot coded distrubution

    A = P(W|R) for W in N
    R = P(AuB |A + B)
    '''

    def __init__(self, agent_a: object, agent_b: object) -> None:
        '''
        Create bounded plane of responses for agent_a & agent_b
        '''
        # init
        self.a: object = agent_a
        self.b: object = agent_b
        self.jaccard_index: float
        self.a_prob_vect: np.array
        self.b_prob_vect: np.array
        self.r: np.ndarray
        self.n: int = len(self.a.response) + len(self.b.response)
        # set values
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
        words_a = set(self.a.response.split())
        words_b = set(self.b.response.split())

        # Calculating the union and intersection of word sets
        union_words = words_a.union(words_b)
        intersection_words = words_a.intersection(words_b)

        # Calculating the Jaccard index
        jaccard_index = len(intersection_words) / \
            len(union_words) if union_words else 0

        # Counting occurrences of each word in the combined responses
        self.r = self.word_freq(self.a.response + self.b.response)
        self.jaccard_index = jaccard_index

        a_prob_vect = self.prob_vect(self.word_freq(self.a.response))
        b_prob_vect = self.prob_vect(self.word_freq(self.b.response))

        self.a_prob_vect = a_prob_vect
        self.b_prob_vect = b_prob_vect

    def prob_vect(self, word_freq_dict: dict()) -> np.array:
        '''
        returns dict of probs of words
        '''
        prob_vect = [0] * len(self.r)

        if self.n == 0:
            self.n = len(self.a.response) + len(self.b.response)

        idx = 0
        for key, _ in self.r.items():
            prob_vect[idx] = word_freq_dict[key.lower()] / self.n
            idx += 1

        return prob_vect

        # for key, value in word_freq_dict:
        #     prob_vect[key] = value / self.n

    def word_freq(self, pasage: str) -> dict:
        '''
        returns a freq dict of words in passage
        '''
        res = defaultdict(int)

        for word in pasage.split():
            res[word.lower()] += 1

        return res


if __name__ == "__main__":
    embedding_params = ["facebook-dpr-ctx_encoder-multiset-base", 200, 25, 0.9]

    agent_ltoa = Agent(
        'agent_ltoa', 'chroma_db/agent_ltoa', 0, embedding_params, True)

    agent_norbert = Agent(
        'agent_norbert', 'chroma_db/agent_norbert', 0, embedding_params, True)

    input_layer = Transducer(agent_ltoa, agent_norbert)
