
import os
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from tqdm.autonotebook import tqdm
from langchain.prompts.prompt import PromptTemplate
from langchain.llms import OpenAI, LlamaCpp
from dataclasses import dataclass
from dotenv import load_dotenv
from abc import ABC, abstractmethod
from constants import CHROMA_SETTINGS, PERSIST_DIRECTORY

load_dotenv()

# @dataclass
# class Keys:

#     #Environment Keys
#     open_api_key: str = os.getenv("OPEN_API_KEY")
#     pinecone_api_key:str  = os.getenv("PINECONE_API_KEY")
#     pinecone_api_env:str = os.getenv("PINECONE_API_ENV")
#     pinecone_index:str = os.getenv("PINECONE_INDEX") 

@dataclass
class Templates:
    #Templates for LLMs
    condense_template:str  = """Given the following conversation and a follow 
    up question, rephrase the follow up question to be a standalone question.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""

    condense_question_prompt:str  = PromptTemplate.from_template(condense_template)

    qa_template = """You are an AI assistant for answering questions about the 
    text or information provided to in the form of pieces of a large text document.
    You are given the following extracted parts of a long document and a question. 
    Provide a conversational answer.
    If you don't know the answer, just say "Hmm, I'm not sure." 
    Don't try to make up an answer.
    If the question is not about the context in the document, 
    politely inform them that you are tuned to only answer questions about the 
    information found in the document."""

    # qa_prompt:str  = PromptTemplate(template=qa_template, 
    #                                 input_variables=["question", "context"])

class LLms(ABC):
    # def __init__(self, keys=None):
    #     self.keys = keys
    
    @abstractmethod
    def get_embeddings(self):
        '''Interface to get Embeddings using the LLM model'''

    @abstractmethod    
    def get_llm(self):
        '''Interface to return LLM Model'''
        

class OpenaiLLms(LLms):
    # def __init__(self, keys:Keys):
    #     self.keys = keys

    def get_embeddings(self) -> OpenAIEmbeddings:
        return OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    def get_llm(self) -> OpenAI:
        return OpenAI(temperature=0, streaming=True, openai_api_key=os.getenv("OPENAI_API_KEY"))

class privateLLms(LLms):
    
    def get_embeddings(self) -> HuggingFaceInstructEmbeddings:
        embeddings_model_name = os.getenv("EMBEDDINGS_MODEL_NAME")
        embeddings = HuggingFaceInstructEmbeddings(model_name=embeddings_model_name, 
                                           model_kwargs={"device": "cpu"})
        return embeddings

    def get_llm(self) -> LlamaCpp:
        model_path = os.getenv("MODEL_PATH")
        model_n_ctx = os.getenv("MODEL_N_CTX")
        max_tokens = os.getenv("MAX_TOKENS")
        llm = LlamaCpp(
            model_path=model_path, 
            n_ctx=model_n_ctx, 
            max_tokens=max_tokens, 
            temperature=0, 
            n_threads=16, 
            repeat_penalty=1.15
            )
        return llm


class Vectorstore:
   
    @staticmethod
    #Load and split the source data 
    def load_and_split_data() -> RecursiveCharacterTextSplitter: 
        if len(os.listdir('data')) > 0:
            loader = DirectoryLoader('data', glob="**/*.pdf")
            raw_data = loader.load()
            print(f"Total number of files are {len(raw_data)}")
        #split all the raw documents
            raw_data_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000, 
                chunk_overlap=0)
            data_chunks = raw_data_splitter.split_documents(raw_data)
            print(f"length of data chunks is {len(data_chunks)}")
        #write data chunks to vectorstore.pkl file
            with open("vectorstore.pkl", "wb") as file:
                pickle.dump(data_chunks, file)
                print("data written to pickle file successfully")
        else:
            raise "No data found, Please upload some Knowledge Base Articles\n"
    
    def as_retriever(self, llm):
        with open("vectorstore.pkl", "rb") as file:
            data_chunks = pickle.load(file)
        #Load data to vectorstore
        embeddings = llm.get_embeddings()
        #import pdb; pdb.set_trace()
        db = Chroma.from_documents(
            data_chunks,persist_directory=PERSIST_DIRECTORY, 
            embedding_function=embeddings, 
            client_settings=CHROMA_SETTINGS
            )
        db.persist()
        retriever = db.as_retriever()
        return retriever

@dataclass
class AllResourcesFactory():

    @property
    def templates(self) -> Templates:
        return Templates
    
    @property
    def llms(self) -> LLms:
        if os.getenv('MODEL_TYPE') == "LLAMA2":
            return privateLLms()
        elif os.getenv('MODEL_TYPE') == "OPENAI":
            return OpenaiLLms()
        else:
            raise "Model Not Supported, choose \"OPENAI\" or \"LLAMA2\""
        
    @property
    def vectorstore(self) -> Vectorstore:
        return Vectorstore()
    
    