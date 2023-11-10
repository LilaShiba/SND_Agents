from typing import List, Union, Optional
import os
import sys
from typing import List, Tuple, Union

import openai
from dotenv import find_dotenv, load_dotenv
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredHTMLLoader
from langchain.document_loaders.csv_loader import CSVLoader

load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']


class NewCourse:
    """ Manages the creation and handling of courses. """

    def __init__(self, name: str, path: str, embedding_params: List[Union[str, float, int]], load_course: bool = False):
        self.name: str = name
        self.doc_path: str = path
        self.docs = None
        self.embedding_function = embedding_params[0]
        self.embedding_params = embedding_params
        if not load_course:
            self.load_documents(path)

    def load_documents(self, path: str) -> Optional[list]:
        try:
            if path.endswith('.pdf'):
                loader = PyPDFLoader(path)
            elif path.endswith('.txt'):
                loader = TextLoader(path)
            elif path.endswith('.csv'):
                loader = CSVLoader(file_path=path)
            elif path.endswith('.html'):
                loader = UnstructuredHTMLLoader(file_path=path)
            else:
                raise ValueError(f"Unsupported file format: {path}")
            self.docs = loader.load()
            return self.docs
        except Exception as e:
            print(f"Error loading documents from {path}: {e}")
            return None

    def from_pdf(self, path: str):
        """
        Loads course content from a PDF file.

        Parameters:
            path (str): Path to the PDF file.

        Returns:
            docs (list): List of documents extracted from the PDF.
        """
        loader = PyPDFLoader(path)
        self.docs = loader.load()
        print('docs created')
        return self.docs

    def from_txt(self, knowledge_document_path):
        """
        Creates new course from txt

        Parameters:
        txt file path

        Returns:
        self.url
        self.docs 
        self.chunks
        self.embeddings

        """
        self.knowledge_document_path = knowledge_document_path
        loader = TextLoader(knowledge_document_path)
        self.docs = loader.load()
        print('docs created')
        return self.docs

    def from_csv(self, knowledge_document_path):
        """
        Creates new course from csv

        Parameters:
        csv file path

        Returns:
        self.url
        self.docs 
        self.chunks
        self.embeddings

        """
        self.knowledge_document_path = knowledge_document_path
        loader = CSVLoader(file_path=knowledge_document_path)
        self.docs = loader.load()
        print('docs created')

    def from_html(self, knowledge_document_path):
        """
        Creates new course from html

        Parameters:
        csv file path

        Returns:
        self.url
        self.docs 
        self.chunks
        self.embeddings

        """
        self.knowledge_document_path = knowledge_document_path
        loader = UnstructuredHTMLLoader(file_path=knowledge_document_path)
        self.docs = loader.load()
        print('docs created')
