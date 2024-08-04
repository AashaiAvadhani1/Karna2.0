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
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage 

#from langchain.memory import ChatMessageHistory
#from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
#from langchain_core.runnables.history import RunnableWithMessageHistory

#print(os.environ.get("ANTHROPIC_API_KEY"))
client_anthropic = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key= os.environ.get("ANTHROPIC_API_KEY"),
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
def develop_privacy_impact_assessment(project):

    data_code = project
    """
    Get the data from the code and classify into sections

    - develop the P1,P2,P3 level classification for each dataset used 
    - whether health data was used 
    - financial or government data used 
    - demographic data (about a user)
    - user account data (passwords, username, recovery questions)

    

    from each classification we can determine the privacy risk per
    then bring up the legislation and statute that the classification will refer t
    """

    prompt = ""
    
    pass



"""
Should return the data classified and why its classified that way
Inputs: Both strings
Proj_data is the information about the project
data_from_code is the data thats used in the project from the code 
"""
def get_data_classification_response(proj_data, data_from_code):
    anthropic_data_classify = """
    Role: Data Classification Privacy Expert

    Get the data from the code and classify into sections
    <context>
    {context_code}
    <context>  
 
    <context>
    {proj_context}
    <context> 

    For each piece of data in the context of the context_code or proj_context do the following:
    Data Classification
        - For every data mentioned in the context_code determine the classification of the data as P2 or not
        - Example of P2 level data are
                    Names
                    Social security numbers (SSN)
                    Addresses
                    Phone numbers
                    Email addresses
                    Financial account details
                    Biometric data

    - whether health data was used 
            - check in the context if any health data was used 
            - biometric data such as x-ray images with individuals names
            - fingerprint data  
    
    - financial or government data used 
        - check if any credit card data is used
        - any billing address or shipping address
        
    - demographic data (about a user)
        - check if any data was used like gender, location, or anything that can identify a user back through the code system. 
    - user account data 
        (passwords, username, recovery questions)


    from each classification we can determine the privacy risk per
    then bring up the legislation and statute that the classification will refer t
    """


    message = client_anthropic.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=4096,
        temperature=0.2,
        system= anthropic_data_classify,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Project Data: " + proj_data + " Code in data: " + data_from_code
                    }
                ]
            }
        ]
    )

    data_assessment_classification = message.content
    print("Anthropic result",data_assessment_classification)

    return data_assessment_classification


def get_data_requirement_documents(legislation_doc, data_from_code):

    embeddings4 = HuggingFaceEmbeddings(model_name='LaBSE')
    #add a section here just for legal documentation
    vector_db = FAISS.load_local("faiss_index", embeddings4, allow_dangerous_deserialization=True)

    prompt_template = """


    Role: Your task is to determine if the data from a piece of code is 

    Gather the types of data that is relevant to the """ + legislation_doc + """ 
    documentation.

    Here is the data from the piece of code.

    Task: Your task is to understand if the data used from the code is mentioned in the legislation document 
    """ + str(data_from_code) + """

    If you are unable to answer the question or pull up relevant information from the legal documents, then answer 
    "Unable to reach a conclusion for the code compliance issue"
    
    """


    retriever_final = vector_db.as_retriever(
    search_type="similarity", search_kwargs={"k": 3} #k: Number of Documents to return, defaults to 4.
    )
    # Define LLM
    model = ChatMistralAI(mistral_api_key= os.environ.get("MISTRAL_API_KEY"), temperature = 0.2)
    # Define prompt template
    prompt = ChatPromptTemplate.from_template("""
    Role: Lawyer Task
    <context>
    {context}
    <context>                          
    Question you must answer: {input}
    You are a lawyer tasked with evaluating the data
                                              

                                              """)

    # Create a retrieval chain to answer questions
    document_chain = create_stuff_documents_chain(model, prompt)
    retrieval_chain = create_retrieval_chain(retriever_final, document_chain)
    response = retrieval_chain.invoke({"context": data_from_code,"input": "List the data that is mentioned in the legislation document " + legislation_doc})


    client_mistral = MistralClient(api_key=os.environ.get("MISTRAL_API_KEY"))
    mistral_model = "mistral-large-latest"

    chat_history = [ChatMessage(role="user", content="Can you give a data classification report on the project?")]
    
    response = client_mistral.chat(
        model=mistral_model, messages=chat_history, tools=tools, tool_choice="auto"
    )

    return response["answer"]


    

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


    project_data_code = project_object.data_from_code



    function_call_data_classification = {
            "type": "function",
            "function": {
                "name": "get_data_classification_response",
                "description": "Gets classification of data used in the model with legislation",
                "parameters": {
                    "type": "string",
                    "properties": {
                        "proj_data": {
                            "type": "string",
                            "description": "Information and Data about the project",
                        },
                        "data_from_code": {
                            "type": "string",
                            "description": "Data that is mentioned in the code of the project",
                        }
                    },
                    "required": ["proj_data, data_from_code"],
                },
            },
        }
    
    names_to_functions = {
        "retrieve_payment_status": functools.partial(retrieve_payment_status, df=df),
        "retrieve_payment_date": functools.partial(retrieve_payment_date, df=df),
    }

    """
    function calling here with this mistral model 
    """
    tools = [function_call_data_classification]

    client_mistral = MistralClient(api_key=os.environ.get("MISTRAL_API_KEY"))
    mistral_model = "mistral-large-latest"

    chat_history = [ChatMessage(role="user", content="Can you give a data classification problem of the project?")]
    
    response = client_mistral.chat(
        model=mistral_model, messages=chat_history, tools=tools, tool_choice="auto"
    )

    print(response)

    return response