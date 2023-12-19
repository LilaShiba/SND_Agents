import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from typing import *
from collections import defaultdict


class MyelinSheath():
    '''
    Takes a Pack of agents &
    constructs a neural network in which each agent
    may share the knodelge is has derived 
    '''

    def __init__(self, pack) -> None:
        self.pack = pack
        '''
        Pack consits of:

        self.edges = list()
        self.weighted_edges = defaultdict()
        self.eigen_central = defaultdict()
        self.G = nx.Graph()
        self.snd: float = 0.00
        self.agents = []
        self.agents_dict = defaultdict()
        '''

    def question(self, user_question: str):
        '''
        wrapper for pack
        ensures neuron representation
        meterics as edges &
        graph representation on agents in pack
        '''
        self.pack.one_question(
            prompt=user_question,
            neuron_representation=True)
        self.pack.update_edges()
        self.pack.graph()

    def agent_transfer(self):
        '''
        A simple BFS for activating the neural network
        '''
        agent_dic = self.pack.agent_dict
        agent_neurons = self.pack.transducer.neurons
        graph = self.pack.G

        # start with most connected node
        eigen_values = nx.eigenvector_centrality(graph)
        node = agent_dic[max(eigen_values, key=eigen_values.get)]
        adj = dict(graph.adjacency())
        # bfs init
        q = [node.name]
        q_name = [node.name]
        # basic bfs for ripple
        while q:
            node = q.pop()
            for edge in adj[node]:
                if edge not in q_name:
                    # update data-structures
                    q.append(edge)
                    q_name.append(edge)
                    print(edge)
                    # processes info via neuron
                    neuron = agent_neurons[edge]
                    print(neuron)
