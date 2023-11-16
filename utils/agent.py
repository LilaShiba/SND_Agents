from typing import *
import numpy as np

from utils.chatbot import ChatBot
from utils.loader import NewCourse
from utils.encoder import Encoder
from utils.knn import Knn


class Agent:
    """
    Represents a top-level AI agent, composed of Encoder, DB, and NewCourse instances.
    """

    def __init__(self, name: str, path: str, cot_type: int, embedding_params: List[Union[str, float, int]], old_course: bool = False):
        """
        Initializes an Agent object with a name, path, type, and embedding parameters.

        Parameters:
            name (str): Name of the agent.
            path (str): Document path.
            cot_type (int): Type of the agent.
            embedding_params (List[Union[str, float, int]]): List containing embedding parameters.
            old_course (bool): True to load a course False to create a course
        """
        self.name: str = name
        self.path: str = path
        self.cot_name: int = cot_type
        self.embedding_params: list = embedding_params
        self.old_course = old_course
        # Pack Details
        self.edges: list = list()
        # input, ouput : x,y for knn is dynamic. Add N Features for knn
        self.state: list = [
            np.random.rand(), np.random.rand(), np.random.rand()]
        self.heading: float = np.random.rand() * 2 * np.pi
        # Subprocesses
        # creates self.docs
        print('🔥  Conjuring up',  self.name,  ' 🔥 ')
        print('')
        print('🧙 creating course  🧙')
        self.course: object = NewCourse(
            name, path, embedding_params, old_course)
        print('')
        print('🔮 creating encoder  🔮 ')
        # creates self.vectordb

        self.encoder: object = Encoder(self.course, old_course)
        print('')
        print('🧚 creating chat_bot for  🧚')
        self.chat_bot: object = ChatBot(self)
        print('')
        print(f'the path  🌈 being used for {self.name} is {path}')
        print('')
        self.vectordb: object = self.encoder.vectordb

    def add_edge(self, node: Any) -> None:
        """
        Add an edge to the object.

        Args:
            node (Any): The node to connect with.
        """
        self.edges.append(node)

    def new_course(self):
        """
        Creates the Docs and Embeddings with a name and path.

        Parameters:
            - name: Name of the agent.
            - path: Document path.
        """
        self.course.from_pdf(self.path)
        self.vectordb = self.encoder.subprocess_create_embeddings(
            self.course.docs)
        print('🌟 instance created 🌟 ')
        print('⚫')

    def start_chat(self):
        """
        Chat with a resource

        Parameters:
            - name: Name of the agent.
            - path: Document path.
        """
        self.chat_bot.load_chat()

    def save(self):
        """
        Save the current instance of Agent to ChromaDB
        DBPath = docs/chroma/self.name

        Parameters:
            - name: Name of the agent.
            - path: Document path.
            - encoder: document instance
            - course: course instance
        """
        self.encoder.subprocess_persist(self.course.knowledge_document_path)
        print(f'instance saved at docs/chroma/{self.name}')

    def load_course(self):
        """
        load vector embeddings from Chroma

        """
        print(f'waking up agent {self.name}')
        print('')

        self.chat_bot.set_agent()

    def load_mem(self):
        """
       load Agent Memory
       Provided self.path is to the DB
       """

    def add_memory(self, path):
        '''
         Add documents to vector db
         path: document path
        '''
        print(f'🦄 adding {path} ')
        print(' ')
        pages = self.course.from_pdf(path)
        docs = self.encoder.create_chunks(pages)

        self.vectordb.add_documents(docs)
        self.vectordb.persist()
        # self.encoder.vectordb.add_documents(embeddings)
        print("memory updated")
        print(' ')

        # with open('output.log', 'r') as file:
        #     # Read the content of the file
        #     pages = self.course.from_txt(res)
        #     docs = self.encoder.create_chunks(pages)
        #     self.chat_bot.add_fractual(docs)

        # Clear the contents of the .log file
        # with open('output.log', 'w') as file:
        #     pass


if __name__ == "__main__":
    embedding_params = ["facebook-dpr-ctx_encoder-multiset-base", 200, 25, 0.7]
    path = 'documents/meowsmeowing.pdf'
    db_path = 'chroma_db/agent_snd'

    testAgent = Agent('agent_snd', path, 0, embedding_params, True)
    # testAgent.add_memory("documents/LtoA.pdf")
    testAgent.start_chat()
