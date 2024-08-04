from typing import List
import os

import streamlit as st
from langchain.vectorstores import FAISS
#from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from src.vector_store import get_conversational_chain
#from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain.chains import RetrievalQA
#from langchain_google_vertexai import VertexAI
from langchain_anthropic import ChatAnthropic
from langchain_google_vertexai import VertexAI
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
from dotenv import load_dotenv

load_dotenv("/Users/aashaiavadhani/Desktop/Karna2.0/src/.env")


os.environ["ANTHROPIC_API_KEY"] = os.environ.get("ANTHROPIC_API_KEY")

def user_input(user_question: str, history: List[dict]):

    embeddings4 = HuggingFaceEmbeddings(model_name='LaBSE')

    file_to_faiss = "/Users/aashaiavadhani/Desktop/Karna2.0/src/faiss_index"
    #faiss_index_dir = FAISS.load_local(file_to_faiss, embeddings4, allow_dangerous_deserialization=True)
    new_db = FAISS.load_local(file_to_faiss, embeddings4, allow_dangerous_deserialization=True)
    #docs = new_db.similarity_search(user_question)
    
    retriever_final = new_db.as_retriever(
            search_type="similarity", search_kwargs={"k": 3} #k: Number of Documents to return, defaults to 4.
        )
    compressor = FlashrankRerank()
    compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever_final
        )
    
    prompt = hub.pull("aavadhan/karnabot", api_url="https://api.hub.langchain.com")
    
    #prompt = hub.pull("rlm/rag-prompt", api_url="https://api.hub.langchain.com")
    role =  """
                Your task is to address relevant questions related to privacy and compliance regulation texts. Adapt your responses to match the style and needs of each question, avoiding unnecessary technical jargon and explaining in simple terms. Do not include information that is unnecessary or irrelevant to the question.

                Identify compliance risks and tasks: Extract the compliance risk from the given text and identify the most important specific compliance tasks to look for during an code evaluation.

                List key factors: List the key factors mentioned in the regulation text.

                Provide citations to the referenced text when asked about specific part of law.

                The four main privacy acts that questions will be about are the CCPA (California Consumer Privacy Act), the HIPAA (Health Insurance Portability and Accountability Act), the GDPR (General Data Protection Regulation), and the Privacy Act of 1974. Provide an overview of the privacy act mentioned in the question.

                Discuss the primary concerns regarding compliance and privacy based on the regulation.

                Highlight potential vulnerabilities: Describe potential vulnerabilities that could lead to non-compliance.

                Present legal text and associated laws: Provide the relevant legal text and the associated compliance law, ensuring the explanation does not exceed 600 words.

                Give examples for specific legislation: If specific legislation is provided, offer examples of situations, code, and practices that could potentially be in violation.

                If prompted to provide example code for compliance and non-compliance, provide code snippet in python.

                List types of data mentioned: Identify the types of data referenced in the legal text.

                Explain your findings clearly and concisely, using paragraph format for questions that require explanation, description, discussion, or comparison. Use bullet points for questions that are specific, require listing, or outlining. Ensure there is natural coherence to the structure of the response where one paragraph semantically connects with the next one.
                
                Question: \n{question}\n

                """

    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20240620",
        temperature=0.2,
        max_tokens=1024,
        timeout=None,
        max_retries=2,
        top_p = 0.4
    )
 
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever= compression_retriever, chain_type_kwargs={"prompt": prompt})
    result = qa_chain({"query": user_question})

    return result["result"]
