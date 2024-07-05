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


def get_conversational_chain():
    prompt_template = """
    If the query contains code, then use this context in the prompt to determine if the code is compliant.

    Your task is to gain the relevant questions towards privacy and compliance regulation texts. 
    Avoid technical jargon and explain it in the simplest of words.

    Abstract: Extract the compliance risk and give three compliance tasks to look for in a code scan
    Give a Relevant Factors, Potential Concerns sections

    Show the legal text and the compliance law that it is associated with (cap at 500 words)
    If given certain legislation, give example of code that could potentially be in violation 

    If you give examples of code, make sure it is only in python 
    List the data that is mentioned legal text
    
    \n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2, top_k=10)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain



def get_code_conversational_chain():
    eu_ai_act_prompt = """
        Role: Lawyer Task

        You are a lawyer tasked with evaluating the data in the EU AI Act legislation. This code can be a piece of software or code building a machine learning or AI model. Your task is to extract and evaluate the data, variables, and types of data used in the code to ensure compliance with the EU AI Act. Follow the steps methodically to ensure we have the correct answer.

        ### Examples of Sensitive Data ###

        - Demographic data
        - Gender
        - Ethnicity
        - Location (zip code)
        - Health information
        - Financial information
        - Personal identification numbers

        ### Steps ###

        1. **Summarize the Objective of the Legislation**: Begin by understanding and summarizing what the legislation is intended to accomplish. Is it performing data analysis, training a machine learning model, or something else?

        2. **Summarize the Data's Purpose in the statute**: Identify the role of the data within the legislation. What is the data used for? How does it contribute to the legislation's objective?

        3. **Explain the Types of Data Used in the legislation**: Identify and explain the types of data used in the legislation. Are they integers, strings, floats, or more complex types?

        4. **Identify Sensitive Data**: Determine if the data includes sensitive information. Look for demographic data, racial data, gender, health information, financial information, etc.

        5. **Assess Compliance with the General Data Protection Regulation Act**: Evaluate whether the usage of sensitive data complies with the General Data Protection Regulation Act. Provide detailed reasoning for your assessment.

        6. **List Variables, Datatypes, Sensitivity, and Compliance**: Create a detailed list of all variables, their corresponding datatypes, whether they are sensitive, and whether their use is compliant.

        7. **Identify Non-Compliant Areas**: Identify and list areas in the code where the data usage does not comply with GDPR and the General Data Protection Regulation Act. Explain why these areas are non-compliant.

        **Example**:
        - `age`: integer, not sensitive, compliant
        - `income`: float, not sensitive, compliant
        - `gender`: string, sensitive, compliant with justification
        - `ethnicity`: string, sensitive, compliant with justification
        - `zip_code`: string, sensitive, compliant
        - `ssn`: string, sensitive, non-compliant because personal identification numbers are not allowed

        ### Example Analysis ###

        1. **Objective**: The legislation aims to train a machine learning model to predict customer churn.
        2. **Data's Purpose**: The data is used to train and validate the machine learning model by providing historical customer data.
        3. **Types of Data**: The legislation uses integers for age, floats for income, and strings for gender and customer ID.
        4. **Sensitive Data**: The legislation includes sensitive data such as gender, ethnicity, and location.
        5. **Compliance with General Data Protection Regulation Act**: The inclusion of sensitive data is justified and complies with the Act's provisions on fairness, transparency, and non-discrimination.
        6. **Variables, Datatypes, Sensitivity, and Compliance**:
        - `age`: integer, not sensitive, compliant
        - `income`: float, not sensitive, compliant
        - `gender`: string, sensitive, compliant with justification
        - `ethnicity`: string, sensitive, compliant with justification
        - `zip_code`: string, sensitive, compliant
        7. **Non-Compliant Areas**: The `ssn` variable is non-compliant because personal identification numbers are not allowed
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2, top_k=10)
    prompt = PromptTemplate(template=eu_ai_act_prompt, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt, document_variable_name="context")
    return chain

"""
Model that gets the data attributes from code 
"""
def get_data_from_model(input_text, model, temperature = 0.3):
    responses = model.generate_content(
        [input_text],
        generation_config={
            "max_output_tokens": 2048,
            "temperature": temperature,
            "top_p": 1
        },
        stream=True,
        
      )
  
    for response in responses:
        print(response.text, end="")
    return responses


def get_code_compliance(code: str) -> str:
    eu_ai_act_prompt = """
    Role: Lawyer Task

    You are a lawyer tasked with evaluating the data in the General Data Protection Regulation Act legislation. This code can be a piece of software or code building a machine learning or AI model. Your task is to extract and evaluate the data, variables, and types of data used in the code to ensure compliance with the General Data Protection Regulation Act. Follow the steps methodically to ensure we have the correct answer.

    ### Examples of Sensitive Data ###

    - Demographic data
    - Gender
    - Ethnicity
    - Location (zip code)
    - Health information
    - Financial information
    - Personal identification numbers

    ### Steps ###

    1. **Summarize the Objective of the Legislation**: Begin by understanding and summarizing what the legislation is intended to accomplish. Is it performing data analysis, training a machine learning model, or something else?

    2. **Summarize the Data's Purpose in the statute**: Identify the role of the data within the legislation. What is the data used for? How does it contribute to the legislation's objective?

    3. **Explain the Types of Data Used in the legislation**: Identify and explain the types of data used in the legislation. Are they integers, strings, floats, or more complex types?

    4. **Identify Sensitive Data**: Determine if the data includes sensitive information. Look for demographic data, racial data, gender, health information, financial information, etc.

    5. **Assess Compliance with the General Data Protection Regulation Act**: Evaluate whether the usage of sensitive data complies with the General Data Protection Regulation Act. Provide detailed reasoning for your assessment.

    6. **List Variables, Datatypes, Sensitivity, and Compliance**: Create a detailed list of all variables, their corresponding datatypes, whether they are sensitive, and whether their use is compliant.

    7. **Identify Non-Compliant Areas**: Identify and list areas in the code where the data usage does not comply with GDPR and the General Data Protection Regulation Act. Explain why these areas are non-compliant.

    **Example**:
    - `age`: integer, not sensitive, compliant
    - `income`: float, not sensitive, compliant
    - `gender`: string, sensitive, compliant with justification
    - `ethnicity`: string, sensitive, compliant with justification
    - `zip_code`: string, sensitive, compliant
    - `ssn`: string, sensitive, non-compliant because personal identification numbers are not allowed

    ### Example Analysis ###

    1. **Objective**: The legislation aims to train a machine learning model to predict customer churn.
    2. **Data's Purpose**: The data is used to train and validate the machine learning model by providing historical customer data.
    3. **Types of Data**: The legislation uses integers for age, floats for income, and strings for gender and customer ID.
    4. **Sensitive Data**: The legislation includes sensitive data such as gender, ethnicity, and location.
    5. **Compliance with General Data Protection Regulation Act**: The inclusion of sensitive data is justified and complies with the Act's provisions on fairness, transparency, and non-discrimination.
    6. **Variables, Datatypes, Sensitivity, and Compliance**:
    - `age`: integer, not sensitive, compliant
    - `income`: float, not sensitive, compliant
    - `gender`: string, sensitive, compliant with justification
    - `ethnicity`: string, sensitive, compliant with justification
    - `zip_code`: string, sensitive, compliant
    7. **Non-Compliant Areas**: The `ssn` variable is non-compliant because personal identification numbers are not allowed
    """

    variables = extract_code_data(code)
    
    user_question = "Does the code use any of the following data types: {}?".format(
        ", ".join([var["type"] for var in variables])
    )
    print("user question", user_question)
    embeddings4 = HuggingFaceEmbeddings(model_name='LaBSE')
    new_db = FAISS.load_local("faiss_index", embeddings4, allow_dangerous_deserialization=True)
    
    docs = new_db.similarity_search(user_question)
    print(docs)
    context = " ".join([doc.page_content for doc in docs])

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)
    prompt = PromptTemplate(template=eu_ai_act_prompt, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt, input_variables=["context", "question"])

    response = chain({"context": context, "question": "Is this code compliant with GDPR and the EU AI Act?"})
    
    return response["output_text"]


def user_input(user_question: str, history: List[dict]):
    embeddings4 = HuggingFaceEmbeddings(model_name='LaBSE')
    new_db = FAISS.load_local("faiss_index", embeddings4, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question, "history": history},
        return_only_outputs=True
    )

    return response["output_text"]






def get_data_code_anthropic(user_code):
    system_prompt = """

Role: Data Scientist

You are given a piece of python code from a developer and this code can be a piece of software or code that is building a machine learning or AI model.
The task is to extract the data, variables, and types of data used in the code.
Lets work this out in a step by step way to be sure we have the right answer.
###  ###

Sensitive Data is defined as the following data:
demographic data, racial data, gender

Insert the privacy data

### Steps ###
1- Summarize the objective of the code
2- Summarize the data's purpose within the code and how its being used
3- Explain what type of data is used in the code whether its integers, strings etc..
4- Identify if the data is sensitive such as demographic data, racial data, gender
5- list out all the variables, its corresponding datatype and whether its sensitive
"""


    """
    Example of text classication for anthropic model 
    
    """
    message = client_anthropic.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=4096,
        temperature=0.2,
        system= system_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_code
                    }
                ]
            }
        ]
    )
    result = message.content
    print("text anthropic: ", type(result[0].text))
    return result[0].text


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
    You are a lawyer tasked with evaluating the data in the General Data Protection Regulation Act legislation. This code can be a piece of software or code building a machine learning or AI model. Your task is to extract and evaluate the data, variables, and types of data used in the code to ensure compliance with the General Data Protection Regulation Act. Follow the steps methodically to ensure we have the correct answer.

    ### Examples of Sensitive Data ###

    - Demographic data
    - Gender
    - Ethnicity
    - Location (zip code)
    - Health information
    - Financial information
    - Personal identification numbers

    ### Steps ###

    1. **Summarize the Objective of the Legislation**: Begin by understanding and summarizing what the legislation is intended to accomplish. Is it performing data analysis, training a machine learning model, or something else?

    2. **Summarize the Data's Purpose in the statute**: Identify the role of the data within the legislation. What is the data used for? How does it contribute to the legislation's objective?

    3. **Explain the Types of Data Used in the legislation**: Identify and explain the types of data used in the legislation. Are they integers, strings, floats, or more complex types?

    4. **Identify Sensitive Data**: Determine if the data includes sensitive information. Look for demographic data, racial data, gender, health information, financial information, etc.

    5. **Assess Compliance with the General Data Protection Regulation Act**: Evaluate whether the usage of sensitive data complies with the General Data Protection Regulation Act. Provide detailed reasoning for your assessment.

    6. **List Variables, Datatypes, Sensitivity, and Compliance**: Create a detailed list of all variables, their corresponding datatypes, whether they are sensitive, and whether their use is compliant.

    7. **Identify Non-Compliant Areas**: Identify and list areas in the code where the data usage does not comply with GDPR and the General Data Protection Regulation Act. Explain why these areas are non-compliant.

    **Example**:
    - `age`: integer, not sensitive, compliant
    - `income`: float, not sensitive, compliant
    - `gender`: string, sensitive, compliant with justification
    - `ethnicity`: string, sensitive, compliant with justification
    - `zip_code`: string, sensitive, compliant
    - `ssn`: string, sensitive, non-compliant because personal identification numbers are not allowed

    ### Example Analysis ###

    1. **Objective**: The legislation aims to train a machine learning model to predict customer churn.
    2. **Data's Purpose**: The data is used to train and validate the machine learning model by providing historical customer data.
    3. **Types of Data**: The legislation uses integers for age, floats for income, and strings for gender and customer ID.
    4. **Sensitive Data**: The legislation includes sensitive data such as gender, ethnicity, and location.
    5. **Compliance with General Data Protection Regulation Act**: The inclusion of sensitive data is justified and complies with the Act's provisions on fairness, transparency, and non-discrimination.
    6. **Variables, Datatypes, Sensitivity, and Compliance**:
    - `age`: integer, not sensitive, compliant
    - `income`: float, not sensitive, compliant
    - `gender`: string, sensitive, compliant with justification
    - `ethnicity`: string, sensitive, compliant with justification
    - `zip_code`: string, sensitive, compliant
    7. **Non-Compliant Areas**: The `ssn` variable is non-compliant because personal identification numbers are not allowed
    
                                              """)

    # Create a retrieval chain to answer questions
    document_chain = create_stuff_documents_chain(model, prompt)
    retrieval_chain = create_retrieval_chain(retriever_final, document_chain)
    response = retrieval_chain.invoke({"context": data_from_code,"input": "List the data that is mentioned in the legislation document " + legislation_doc})
    return response["answer"]



def get_result(legal_doc_data_requirements, data_code):

    decider_prompt = """
            You are an evaluator of compliance.
            You are given 2 pieces of context, a piece of code and the sensitive data mentioned in a legal documentation.

            If the data that is used in the code is similar to the data mentioned in the legal documentation (that is sensitive and what should not be used)
            then the code has evidence to be non compliant and you have to explain why.
                Explain the reason why if the data is mentioned then why it should be not compliant with GDPR. 
                Mention the data that is used and mention the statue and text that the legislation came from. 


            If the data that is used in the code is not similar to the sensitive data mentioned in the legal documentaiton, 
            then the code has evidence to be compliant and explain why 
                Explain the reason why if the data is not mentioned then why it should be more compliant with GDPR. 

                
            Give a confidence score from 0 to 100 percent on your conclusion.

            ALWAYS give a warning saying to consult a expert privacy counsel and this is not legal advice 
             """

    print("legal doc",type(legal_doc_data_requirements))
    print("data code",type(data_code))


    message = client_anthropic.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=4096,
        temperature=0.2,
        system= decider_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Legal Text Data: " + legal_doc_data_requirements + " Code in data: " + data_code
                    }
                ]
            }
        ]
    )

    compliance_assessment = message.content
    print("Anthropic result",compliance_assessment)
    return compliance_assessment[0].text

def check_code_compliance(user_code: str):
    
    # get the data extracted from the code
    data_code = get_data_code_anthropic(user_code)
    # get the data requirement from the legal doc

    legal_doc = "General Data Protection Regulation"
    legal_doc_data_requirements = get_data_requirement_documents(legal_doc,data_code)
    
    result_code_compliance = get_result(legal_doc_data_requirements, data_code)
    
    return result_code_compliance
