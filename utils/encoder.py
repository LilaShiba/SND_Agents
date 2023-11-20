from typing import *
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import Chroma


class Encoder:
    """
    Handles the encoding of documents for a course.
    """

    def __init__(self, course_instance: object, load: bool = False):
        """
        Initializes an Encoder with a course instance.

        Parameters:
            course_instance (object): Instance of NewCourse.
        """
        self.model: str = course_instance.embedding_params[0]
        self.chunk_size: int = course_instance.embedding_params[1]
        self.overlap: int = course_instance.embedding_params[2]
        self.course_instance = course_instance
        self.name = course_instance.name

        self.docs = course_instance.docs
        self.chunks = None
        self.vectordb = None
        self.k_results = None
        self.embedding_check = False

        self.path_to_db = None

        self.embedding_function = SentenceTransformerEmbeddings(
            model_name=self.model)
        if load:
            self.from_db(self.course_instance.doc_path)
        else:
            self.subprocess_create_embeddings()

    def create_chunks(self, chunk=200, overlap=15):
        """
        Creates new chunks from documents
        """

        text_splitter = TokenTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.overlap)

        self.chunks = text_splitter.split_documents(self.docs)
        return self.chunks

    def embed_chunks(self, persist_directory='docs/chroma/'):
        """
        Creates new embeddings from chunks

        Parameters:
        persist_directory path
        self.docs
        self.chunks

        Returns:
        self.vectordb
        """

        embedding = OpenAIEmbeddings()
        self.vectordb = Chroma.from_documents(
            documents=self.chunks,
            embedding=embedding,
            persist_directory=persist_directory
        )
        self.path_to_db = persist_directory

    def subprocess_create_embeddings(self, docs=None):
        """
        Creates new course from processed docs

        Parameters:
        self.docs

        Returns:
        self.chunks
        self.embeddings
        self.vectordb

        """
        if not docs:
            docs = self.docs
        self.create_chunks(docs)
        print("🧩 chunks created for", self.name)
        print('')
        self.embed_chunks()
        print(" 🔗 embedding created for", self.name)
        print('')
        # save to disk
        self.vectordb = Chroma.from_documents(
            self.docs, self.embedding_function, persist_directory="./chroma_db/"+self.name)
        self.course_instance.vectordb = self.vectordb
        self.vectordb.persist()
        self.path_to_db = "./chroma_db/"+self.name
        return self.vectordb

    def subprocess_persist(self, path, model="facebook-dpr-ctx_encoder-multiset-base"):
        """
        Creates new course docs, chunks, and embeddings 
        from processed docs

        Parameters:
        self.docs

        Returns:
        self.chunks
        self.embeddings
        self.vectordb

        """
        embedding_function = SentenceTransformerEmbeddings(
            model_name=model)

        self.create_chunks(self.docs)

        # save to disk
        vectordb = Chroma.from_documents(
            self.chunks, embedding_function, persist_directory=path)
        vectordb.persist()
        self.vectordb = vectordb
        self.embedding_function = embedding_function
        self.course_instance.vectordb = self.vectordb
        self.path_to_db = path

    def encoded_query(self, q_a, k_docs=5):
        """
       Encodes query then searches

        Parameters:
        self.docs

        Returns:
        k docs
        doc len

        """
        if self.course_instance.cot:
            cot_q = "step by step and one by one explain" + q_a
            docs = self.vectordb.similarity_search_with_score(
                query=cot_q, distance_metric="cos", k=k_docs)
        else:
            docs = self.vectordb.similarity_search_with_score(
                query=q_a, distance_metric="cos", k=k_docs)
        return docs, len(docs)

    def from_db(self, path_to_db, model="facebook-dpr-ctx_encoder-multiset-base"):
        """
        loads vector embeddings for Agent
        Start of memory 
        TODO: Add to pipeline
        """
        model = "facebook-dpr-ctx_encoder-multiset-base"

        embedding_function = SentenceTransformerEmbeddings(
            model_name=model)

        print('loading', self.name, '...')

        self.vectordb = Chroma(persist_directory=path_to_db,
                               embedding_function=embedding_function)
        print('agent', self.name, 'loaded')

    def add_documents(self, path):
        '''
        Add documents to vector db
        '''

        self.vectordb.add_documents(documents=path)
        self.vectordb.persist()

    def get_embeddings(self, docs, path_to_db):
        """
        Add documents
        """
        embedding = OpenAIEmbeddings()
        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=embedding,
            persist_directory=path_to_db
        )
        return vectordb
