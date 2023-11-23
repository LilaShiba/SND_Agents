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

    def __init__(self, agents: list, question: str) -> None:
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
        # set values
        for agent in agents:
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
            prob_vect[idx] = word_freq[key.lower()] / self.n
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


if __name__ == "__main__":
    embedding_params = ["facebook-dpr-ctx_encoder-multiset-base", 200, 25, 0.9]

    agent_ltoa = Agent(
        'agent_ltoa', 'chroma_db/agent_ltoa', 0, embedding_params, True)

    agent_norbert = Agent(
        'agent_norbert', 'chroma_db/agent_norbert', 0, embedding_params, True)

    input_layer = Transducer(
        [agent_ltoa, agent_norbert], "how can one design a neuron?")
