from utils.agent import Agent
from utils.metrics import ThoughtDiversity
from utils.connections import Knn

from typing import Any
import logging
from itertools import combinations
from collections import defaultdict
import time
import networkx as nx
import matplotlib.pyplot as plt


class Pack:
    '''
    three agent instances are composed as one
    '''

    def __init__(self, agent_specs: list, embedding_params: list = None) -> object:
        '''
        Create Pack
        Default is of three DPR Corpus agents

        agent_specs (list): name, path, cot_type, new_bool
        embedding_params (list): llm, chunk, overlap, creativeness
        '''
        self.edges = list()
        self.weighted_edges = defaultdict()
        self.eigen_central = defaultdict()
        self.G = nx.Graph()
        self.agents = []
        self.agents_dict = defaultdict()
        if not embedding_params:
            embedding_params = {
                1: ["facebook-dpr-ctx_encoder-multiset-base", 200, 25, 0.9],
                2: ["facebook-dpr-ctx_encoder-multiset-base", 200, 25, 0.5],
                3: ["facebook-dpr-ctx_encoder-multiset-base", 200, 25, 0.1]
            }

        # subprocess add agents
        for idx, _ in enumerate(agent_specs):
            name, path, cot_type, new_bool, = agent_specs[idx]
            delta_agent = Agent(name, path, cot_type,
                                embedding_params[idx], new_bool)
            self.agents_dict[name] = delta_agent
            self.agents.append(delta_agent)
        self.agent_names = [agent.name for agent in self.agents]
        # init basic random network structure
        self.edge_algos = Knn(self.agents)
        self.agent_dict = {agent.name: agent for agent in self.agents}
        self.current_res = None
        self.current_jaccard_indices = None
        self.embeddings = embedding_params
        # Create or load embeddings
        self.load_agent_docs()
        self.metrics = ThoughtDiversity(self)

    def update_weighted_edges(self, question: str, k: int = 0) -> dict:
        '''
        cycle through knn to add edges for K combinations
        Within range x
        TODO: get distrubution rather than last set in montecarlo expirment
        '''
        edges = defaultdict()
        # self.metrics.monte_carlo_sim(question=question, rounds=1)
        self.one_question(prompt=question)
        print('monte carlo finished')
        for idx, node in enumerate(self.agents):
            delta_edges = self.edge_algos.search(node.state, k)
            self.agents[idx].edges.append([n.name for n in delta_edges])
            edges[node.name] = [
                n.name for n in delta_edges if n.name != node.name]

        self.edges = edges
        return edges

    def update_edges(self, question: str = "", k: int = 0) -> dict:
        '''
        cycle through knn to add edges for K combinations
        Within range x

        '''
        edges = defaultdict()
        # self.one_question(prompt=question)
        if question != "":
            self.metrics.set_vectors(question)
        for idx, node in enumerate(self.agents):
            delta_edges = self.edge_algos.search(node.state, k)
            self.agents[idx].edges.append([n.name for n in delta_edges])
            edges[node.name] = [node.name for node in delta_edges]

        self.edges = edges
        return edges

    def graph(self):
        '''
        creates networkX graph from self.edges
        '''
        self.G = nx.Graph()
        for agent, connections in self.edges.items():
            for connection in connections:
                # Assuming each connection is a tuple representing another agent

                self.G.add_edge(agent, str(connection))

        # Now, draw the graph
        plt.figure(figsize=(12, 8))  # Set the size of the plot
        pos = nx.spring_layout(self.G)  # This is a layout for the nodes

        # Draw the graph with labels
        nx.draw(self.G, pos, with_labels=True, node_color='lightblue',
                edge_color='gray', node_size=200, font_size=10)

        # Display the plot
        plt.show()
        self.eigen_central = nx.eigenvector_centrality(self.G)
        print(self.eigen_central)
        return self.G

    def load_agent_docs(self):
        '''
        Load or Create embeddings for Agent_Name at DB_Path

        '''
        idx = 0
        for agent in self.agents:
            # New Instance
            if not agent.old_course:
                agent.new_course()

            # Load VectorDB
            else:
                agent.load_course()
            idx += 1

    def add_docs(self, docs: list):
        '''
        update all agents with list/path of document(s)
        '''

        if isinstance(docs, list):
            print('meow')
            for doc in docs:
                for delta_agent, _ in self.agents:
                    delta_agent.chat_bot.add_new_docs(doc)
        else:
            print('bork')
            for delta_agent in self.agents:
                delta_agent.chat_bot.add_fractual(docs)

        print('upload successful :)')

    def one_question(self, prompt: str):
        '''
        one question for pack
        '''
        res = defaultdict()
        for agent in self.agents:
            res[agent.name] = agent.chat_bot.one_question(prompt)
            # time.sleep(60)
        logging.info(res)
        # logging.debug(res)
        return res

    def chat(self):
        '''
        Speak with all agents at one time
        '''
        exit_flag = False
        res = defaultdict()

        while not exit_flag:
            # Get question
            prompt = input("please ask a question to the pack")
            # Exit Path
            if 'exit' in prompt:
                exit_flag = True

            for agent in self.agents:
                res[agent.name] = agent.chat_bot.one_question(prompt)
                # `time.sleep(60)
            logging.info(res)
            print('Here are the collective answers: ')
            print(res)
            self.current_res = res
            # self.jaccard_similarity(res)


if __name__ == '__main__':

    embedding_params = [
        ["facebook-dpr-ctx_encoder-multiset-base", 200, 25, 0.9],
        ["facebook-dpr-ctx_encoder-multiset-base", 150, 20, 0.7],
        ["facebook-dpr-ctx_encoder-multiset-base", 100, 15, 0.5]
    ]
    # name, path, cot_type, new_bool
    agent_specs = [
        ['agent_ltoa', 'documents/LtoA.pdf', 1, 1],
        ['agent_snd', 'documents/SND.pdf', 1, 1],
        ['agent_foundation', 'documents/meowsmeowing.pdf', 1, 1]
    ]

    test_pack = Pack(agent_specs, embedding_params)

    test_pack.add_docs(['documents/SND.pdf'])
    # test_agent.chat()
    metrics = ThoughtDiversity(test_pack)
    print(metrics.monte_carlo_sim())
