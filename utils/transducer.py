import numpy as np
import matplotlib.pyplot as plt
from typing import *
from utils.agent import Agent
from utils.nets import Neuron
from collections import defaultdict, Counter


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
        self.agents = {agent.name: agent for agent in agents}

        self.jaccard_indices = defaultdict(float)
        self.prob_vectors = defaultdict(list)
        self.all_words_count = defaultdict(int)
        self.all_responses: str = ""
        self.n: int
        self.neurons = defaultdict()
        # set values

        for agent in agents:
            if question != "":
                print('loading agent', agent.name, '... ')
                agent.chat_bot.one_question(question)
            self.all_responses += agent.response

        self.all_responses = self.all_responses.split()
        self.n = len(self.all_responses)
        print(self.all_responses)
        self.agent_response = {agent.name: agent.response.split()
                               for agent in agents}
        self.all_words_count = self._get_word_freq(self.all_responses)
        self.all_words_prob = self._get_prob_vect(self.all_responses)
        self.prob_vectors = {agent.name: [
            self._get_prob_vect(agent.response)] for agent in agents}

        # self.bounded_space()

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
            response = agent.response.split()
            # Calculating the union and intersection of word sets
            union_words = response.union(self.all_responses)
            intersection_words = response.intersection(self.all_responses)

            # Calculating the Jaccard index
            jaccard_index = len(intersection_words) / \
                len(union_words) if union_words else 0

            self.jaccard_indices[name] = jaccard_index

            # for name_2, agent_2 in self.agents.items():
            #     if name_2 != name:
            #         union_words = response.union(agent_2.response)
            #         intersection_words = response.intersection(self.all_responses)

            #         # Calculating the Jaccard index
            #         jaccard_index = len(intersection_words) / \
            #             len(union_words) if union_words else 0

            #         self.prob_vectors[(name, name_2)] =
            #         self.jaccard_indices[(name, name_2)] =

            # Counting occurrences of each word in the combined responses
            # self.prob_vectors[name] = self.prob_vect(response)

    def _get_prob_vect(self, response: str):
        '''
        input:
        set of response string
        returns:
        prob-vector: (np.array) word_count / total_response_words
        hot-coded: (np.array) word_count / total_same_word_in_all_responses
        '''
        if isinstance(response, str):
            response = response.split()

        agent_w_count = self._get_word_freq(response)
        hotcode_vector = [0] * len(self.all_responses)
        probabilty_vector = [0] * len(agent_w_count)
        hotcode_vector = np.array(
            [agent_w_count[key.lower()] / self.n for key in self.all_responses])
        probabilty_vector = np.array([agent_w_count[key.lower(
        )] / self.all_words_count[key.lower()] for key in self.all_responses])

        return [probabilty_vector, hotcode_vector]

        # for key, value in word_freq_dict:
        #     prob_vect[key] = value / self.n

    def _get_jaccard(self, agent):
        response = agent.response.split()
        union_words = response.union(self.all_responses)
        intersection_words = response.intersection(self.all_responses)

        # Calculating the Jaccard index
        jaccard_index = len(intersection_words) / \
            len(union_words) if union_words else 0
        self.jaccard_indices[agent.name] = jaccard_index

    def _get_word_freq(self, pasage: str) -> dict:
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
        counts = Counter(response)
        n = len(response)
        probs = [c/n for c in counts.values()]
        return -sum(p * np.log(p) if p > 0 else 0 for p in probs)

    def true_diversity(self, response: str) -> float:
        """
        Calculates the True Diversity (D) using Shannon Entropy (H).

        Parameters:
        counts (List[float]): A list of probabilities.

        Returns:
        float: True Diversity of the dataset.
        """
        return np.exp(self.shannon_entropy(response))

    def create_layer(self) -> np.array:
        '''
        creates transducer layer for input
        '''

        main_vector = self.all_words_prob
        idx = 0
        for name, agent in self.agents.items():
            # hot key-encode vector
            prob_v = self.prob_vectors[name][0][1]
            delta_dot = np.dot(main_vector, prob_v)
            print(f'delta dot product {delta_dot}')
            self.neurons[name] = Neuron(input=delta_dot[1], id=idx, layer=1)
            self.neurons[name].probabilty_vector = prob_v
            self.neurons[name].dot_product = delta_dot
            self.neurons[name].cross_product = main_vector * prob_v
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
