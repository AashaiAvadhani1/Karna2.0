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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
import joblib
from dotenv import dotenv_values


config = dotenv_values(".env")  # config = {"USER": "foo", "EMAIL": "foo@example.org"}

DATA_DIR = 'data/'


client_anthropic = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key= "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",
)


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

List all the data mentioned step by step including variable names, the corresponding data type and a short 5 sentence description of the data

### Steps ###
1- Summarize the objective of the code
2- Summarize the data's purpose within the code and how its being used
3- Explain what type of data is used in the code whether its integers, strings etc..
4- Identify if the data is sensitive such as demographic data, racial data, gender
5- Classify each data mentioned in the code according to the privacy classification above 
6- list out all the variables, its corresponding datatype and whether its sensitive
7. List which systems are used and if you see any data storage transfers
8. 
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



def generate_pia_answers(pia_template, data_code):

    decider_prompt = """

            FOR EACH SECTION MENTION THE DATA USED AND CODE PER SECTION 
            Look at the way the data is being used.

            Generate a table and classify the data used 

            If it is possible to identify a user based off the data , then that is a compliance risk and should be mentioned throughout the PIA template. 

            Mention if there is any hashing or P1 or P2 level data used

            ALWAYS GIVE A TEMPLATE WITH THE 9 SECTIONS
             You are an evaluator of compliance. You are given a privacy impact assessmente template.

            Your task is to look at the template and fill out each section to the best of your ability given the information about the project code.
        
            ALWAYS give a warning saying to consult a expert privacy counsel and this is not legal advice 

            The output should be a 9 section document about a PIA given the code requirements. 
             
            Show the developer how to improve the software project so that is has less privacy risks 



             """

    print("legal doc",type(pia_template))
    print("data code", data_code)


    message = client_anthropic.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=4096,
        temperature=0.1,
        system= decider_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Legal Text Data: " + pia_template + " Code in data: " + data_code
                    }
                ]
            }
        ]
    )

    compliance_assessment = message.content
    print("Anthropic result",compliance_assessment)
    return compliance_assessment[0].text





def get_pia_template():
    pia_template = """
### Privacy Impact Assessment (PIA) Template


FOR EACH SECTION MENTION THE DATA USED AND CODE PER SECTION 


ANSWER 

---

#### Section 1: Project Information
1. **Project Name**:
2. **Project Description**:
3. **Project Lead**:
4. **Contact Information**:
5. **Date**:

---

#### Section 2: Data Collection and Processing
1. **Describe the personal data to be collected**:
   - Types of data (e.g., name, address, email)
   - Special categories of data (e.g., health information, financial data)

2. **Purpose of data collection**:
   - Why is this data being collected?
   - How will it be used?

3. **Data sources**:
   - Where will the data come from (e.g., directly from individuals, third parties)?

---

If it is possible to identify a user based off the software project should be mentioned here
#### Section 3: Legal and Compliance
1. **Legal basis for processing**:
   - Consent
   - Contractual necessity
   - Legal obligation
   - Vital interests
   - Public task
   - Legitimate interests

2. **Relevant laws and regulations**:
   - GDPR
   - CCPA
   - Other applicable laws

---

#### Section 4: Data Storage and Security
1. **Data storage location**:
   - mention the data used and where its stored
   - Where will the data be stored?
   - Will it be stored locally or in the cloud?

2. **Security measures in place**:
   - Encryption
   - Access controls
   - Data backup procedures

---

#### Section 5: Data Sharing and Transfers


1. **Data sharing**:
    - For this section look at what data the code is using and classify it as p1 p2 level data
    - see if the data is being shared to different sources
   - Who will the data be shared with?
   - For what purpose?



2. **Data transfers**:
   - Will the data be transferred outside the organization?
   - Will it be transferred internationally?
   - What safeguards are in place for international transfers?
   - Answer the following questions 
   - transfer of data is seen when the code is moving data between contains or systems like Google to AWS

---

#### Section 6: Data Retention and Deletion
1. **Data retention policy**:
   - How long will the data be kept?
   - What criteria will be used to determine the retention period?

2. **Data deletion**:
   - What processes are in place for secure data deletion?

---

#### Section 7: Risk Assessment and Mitigation
1. **Potential risks to data subjects**:
   - Unauthorized access
   - Data breaches
   - Inaccurate data

2. **Risk mitigation measures**:
   - Measures taken to mitigate identified risks
   - Any remaining risks and how they will be managed

---


 Add more information and guarantee that the thing is there
#### Section 8: Data Subject Rights
1. **Rights of data subjects**:
   - Access
   - Rectification
   - Erasure
   - Restriction
   - Portability
   - Objection

2. **Procedures for handling data subject requests**:
   - How can individuals exercise their rights?
   - What is the process for handling these requests?

---

#### Section 9: Consultation and Approval
1. **Consultation with stakeholders**:
   - Have stakeholders been consulted about this PIA?
   - Who was consulted?

2. **Approval**:
   - Project Lead signature
   - Date
   - Data Protection Officer (DPO) signature (if applicable)
   - Date

---

This template can be customized to suit the specific needs and requirements of your organization. Ensure that each section is thoroughly reviewed and completed before finalizing the PIA.
"""

    return pia_template


def generate_pia(user_code: str):
    data_code = get_data_code_anthropic(user_code)
    privacy_impact_assessment_template = get_pia_template()
    pia_assessment = generate_pia_answers(privacy_impact_assessment_template, data_code)
    return pia_assessment