from typing import List

from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate

from langchain.vectorstores import FAISS
from langchain_google_genai import (ChatGoogleGenerativeAI,
                                    GoogleGenerativeAIEmbeddings)
from typing import List
from langchain.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.memory import ChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
import anthropic 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

client_anthropic = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key="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",
)



"""
DEPRIORITIZE: NOT CALLED which means we have to change the system....
Creaitng a multi turn system for past messages 
"""
def create_memory_chain(llm, base_chain, chat_memory):
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    runnable = contextualize_q_prompt | llm | base_chain

    def get_session_history(session_id: str) -> ChatMessageHistory:
        return chat_memory

    with_message_history = RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )
    return with_message_history



"""
Description: Gets data used from the code snippet submitted by user
"""
def get_data_code_anthropic(user_code):
    system_prompt = """
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

5. **Assess Compliance with the EU AI Act**: Evaluate whether the usage of sensitive data complies with the EU AI Act. Provide detailed reasoning for your assessment.

6. **List Variables, Datatypes, Sensitivity, and Compliance**: Create a detailed list of all variables, their corresponding datatypes, whether they are sensitive, and whether their use is compliant.

7. **Identify Non-Compliant Areas**: Identify and list areas in the code where the data usage does not comply with GDPR and the EU AI Act. Explain why these areas are non-compliant.

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
5. **Compliance with EU AI Act**: The inclusion of sensitive data is justified and complies with the Act's provisions on fairness, transparency, and non-discrimination.
6. **Variables, Datatypes, Sensitivity, and Compliance**:
- `age`: integer, not sensitive, compliant
- `income`: float, not sensitive, compliant
- `gender`: string, sensitive, compliant with justification
- `ethnicity`: string, sensitive, compliant with justification
- `zip_code`: string, sensitive, compliant
7. **Non-Compliant Areas**: The `ssn` variable is non-compliant because personal identification numbers are not allowed.
"""
    prompt = f"\n\nHuman: {system_prompt}\n\nAssistant:"

    response = client_anthropic.completions.create(
        model="claude-3-opus-20240229",
        max_tokens=4096,
        temperature=0.0,
        prompt=prompt + user_code
    )

    result = response['choices'][0]['text']
    print(result)
    return result


def get_conversational_chain():
    """Returns a conversational chain."""

    prompt_template = """
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
    
    \n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
 
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5, top_k = 10)

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain




"""
Re-ranking llama index and cohere part 
"""

"""
Gets called for the user input and legal vectordb
"""
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
