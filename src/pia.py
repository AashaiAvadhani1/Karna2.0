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

mistral_model = ChatMistralAI(mistral_api_key= os.environ.get("MISTRAL_API_KEY"), temperature = 0.5)

"""
Description: This is for generating PIA questions NOT Answering
From the PIA embedding store,
 we can query into the embedding to get the templates for a Privvacy impact assessments
"""
def get_template_PIA():
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
Get some fucntion calls in here per section of the PIA 

Data  Collection (Data Classification)


Data Subjects


Data Usage


Data Processing, transfers, storage


Analytics, Profile, Automated Decision Making


Security and logging


Third Party Data Processing


Individual Rights


Marketing

"""



"""
Easier dev time
This will generate a privacy impact assessment from scratch!
Use the template from onetrust
"""
def develop_privacy_impact_assessment(comp_name, proj_name, project_description, data_dictionary):
    pass



"""
Should return the data classified and why its classified that way
"""
def get_data_collection_response():
    

    pass

def get_data_subject_response():
    pass

def get_data_usage_response():
    pass

def get_data_process_storage_response():
    pass



"""
Harder dev time (per function call agent category)
Will answer a PIA question individually per query from a legal user
project_object is the code_class
"""
def answer_pia_questions_individually(project_object,privacy_query):
    print(project_object.display_info())
    #get the data from the code (should be stored as an object)
    project_data_code = project_object.data_from_code
    #per function call for each category, build the agent here for answering PIA questions
    #Per cateogry have to answer each PIA question
    data_collection_response = {
            "type": "function",
            "function": {
                "name": "retrieve_payment_date",
                "description": "Gets classification of data used in the model with legislation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "merch_name": {
                            "type": "string",
                            "description": "Text Classification",
                        }
                    },
                    "required": ["project_description, data_dictionary"],
                },
            },
        }
    
    """
    function calling here with this mistral model 
    """
    tools = [tool_payment_status, tool_payment_date]

    response = client.chat(
        model=mistral_model, messages=chat_history, tools=tools, tool_choice="auto"
    )
    print(data_collection_response)

    #have the mistral model here pick the function

    pass