import os
from typing import List, Dict, Any
#from langchain import HuggingFaceEmbeddings, FAISS, ChatGoogleGenerativeAI, PromptTemplate, load_qa_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate

from langchain.vectorstores import FAISS
from langchain_google_genai import (ChatGoogleGenerativeAI,
                                    GoogleGenerativeAIEmbeddings)
from typing import List
from langchain.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
import anthropic

## Installation of MISTRAL
from langchain_community.document_loaders import TextLoader
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

#from langchain.memory import ChatMessageHistory
#from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
#from langchain_core.runnables.history import RunnableWithMessageHistory

client_anthropic = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key= "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",
)

mistral_model = ChatMistralAI(mistral_api_key= os.environ.get("MISTRAL_API_KEY"), temperature = 0.6)

"""
From the PIA embedding store, we can query into the embedding to get the templates for a Privvacy impact assessments
"""
def get_template_PIA():
    user_question = ""
    embeddings4 = HuggingFaceEmbeddings(model_name='LaBSE')
    new_db = FAISS.load_local("faiss_index_pia", embeddings4, allow_dangerous_deserialization=True)

    """
    Use chain of thought to develop 50 questions from the particular
    response of the LLM when querying into the PIA template document 

    Use strict prompting in order to get the various sections of a privacy impact assessment
    """
    retriever_final = new_db.as_retriever(
    search_type="similarity", search_kwargs={"k": 3} #k: Number of Documents to return, defaults to 4.
    )
    # Define LLM
    model = ChatMistralAI(mistral_api_key= os.environ.get("MISTRAL_API_KEY"), temperature = 0.2)
    # Define prompt template
    prompt = ChatPromptTemplate.from_template("""
    Role: Privacy Program Manager

    Your task is to generate questions from a Privacy Impact Assessments
    
    """)

    # Create a retrieval chain to answer questions
    document_chain = create_stuff_documents_chain(model, prompt)
    retrieval_chain = create_retrieval_chain(retriever_final, document_chain)
    response = retrieval_chain.invoke({"input": "Generate a couple questions about  "})
    return response["answer"]


"""
Based on a user query, the llm system should answer the pia question
"""
def answer_pia_questions(user_question):
    


    pass