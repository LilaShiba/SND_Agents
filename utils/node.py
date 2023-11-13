import numpy as np
from typing import *


class Node:

    '''
    represent an agent as a node
    in a neural network
    '''

    def __init__(self, agent: object) -> object:
        self.state = agent.state
        self.vector = agent.vector
        self.heading = agent.heading
