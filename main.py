import os
import joblib
import streamlit as st
from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv
import google.generativeai as genai
from src.user_interface import user_input
from src.vector_store import *
from src.code_compliance import *
from src.pia import *
from src.code_class import *
from helper import *
import fitz  # PyMuPDF for PDF processing
import time

#Welcome to KarnaBot

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Constants
MODEL_ROLE = 'ai'
AI_AVATAR_ICON = '✨'
DATA_DIR = 'data/'

# Create data/ directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

# Load past chats
def load_past_chats():
    try:
        return joblib.load(os.path.join(DATA_DIR, 'past_chats_list'))
    except:
        return {}

past_chats = load_past_chats()

# Function to chunk text
def get_text_chunks(text, chunk_size=1000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Function to save chunks to a file
def save_chunks(chunks, chunk_file):
    joblib.dump(chunks, chunk_file)

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function for the questionnaire page
def questionnaire():
    st.header("KarnaAutomate")
    st.caption("Get your PIA done in less than 10 minutes")
    
    # Example questions
    company_name = st.text_input("Company Name:")
    project_name = st.text_input("Project Name:")
    additional_notes = st.text_area("Enter a Data Dictionary, Project Description etc...")
    code = st.text_area("Enter the Code for your project:")

    uploaded_file = st.file_uploader("Or Upload your documents as a PDF", type=["pdf"])
    if uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            text = extract_text_from_pdf(uploaded_file)
            st.text_area("Extracted Text", value=text, height=300)
            # Process the text if needed, e.g., save to a file, analyze, etc.
            chunks = get_text_chunks(text)
            save_chunks(chunks, os.path.join(DATA_DIR, 'pdf_document_chunks.pkl'))
            st.success("PDF processed and text extracted successfully!")

    if st.button("Submit"):
        st.success("Questionnaire submitted successfully!")
        # Save the responses if needed
        responses = {
            "Company Name": company_name,
            "Project Name": project_name,
            "Code": code,
            "Additional Notes": additional_notes,
            "Uploaded File": uploaded_file.name if uploaded_file else None
        }
        joblib.dump(responses, os.path.join(DATA_DIR, 'questionnaire_responses.pkl'))
        return responses

# Main function
def main():
    st.set_page_config("KarnaBot")

    # Sidebar navigation
    page = st.sidebar.selectbox("Choose a page", ["Chat", "Questionnaire"])

    if page == "Chat":
        # Sidebar for past chats
        with st.sidebar:
            st.write('# Past Chats')
            if st.session_state.get('chat_id') is None:
                st.session_state.chat_id = st.selectbox(
                    label='Pick a past chat',
                    options=[f'{time.time()}'] + list(past_chats.keys()),
                    format_func=lambda x: past_chats.get(x, 'New Chat'),
                    placeholder='_',
                )
            else:
                st.session_state.chat_id = st.selectbox(
                    label='Pick a past chat',
                    options=[f'{time.time()}', st.session_state.chat_id] + list(past_chats.keys()),
                    index=1,
                    format_func=lambda x: past_chats.get(x, 'New Chat' if x != st.session_state.chat_id else st.session_state.chat_title),
                    placeholder='_',
                )
            st.session_state.chat_title = f'ChatSession-{st.session_state.chat_id}'

        # Load or initialize chat history
        try:
            st.session_state.messages = joblib.load(
                f'{DATA_DIR}/{st.session_state.chat_id}-st_messages'
            )
            st.session_state.gemini_history = joblib.load(
                f'{DATA_DIR}/{st.session_state.chat_id}-gemini_messages'
            )
        except:
            st.session_state.messages = []
            st.session_state.gemini_history = []

        st.session_state.model = genai.GenerativeModel('gemini-pro')
        st.session_state.chat = st.session_state.model.start_chat(
            history=st.session_state.gemini_history,
        )

        # Display chat messages from history
        for message in st.session_state.messages:
            with st.chat_message(
                name=message['role'],
                avatar=message.get('avatar'),
            ):
                st.markdown(message['content'])

        # User input
        if prompt := st.chat_input('Your message here...'):
            if st.session_state.chat_id not in past_chats.keys():
                past_chats[st.session_state.chat_id] = st.session_state.chat_title
                joblib.dump(past_chats, os.path.join(DATA_DIR, 'past_chats_list'))

            with st.chat_message('user'):
                st.markdown(prompt)
            st.session_state.messages.append(
                dict(
                    role='user',
                    content=prompt,
                )
            )

            # Generate response
            response = user_input(prompt, st.session_state.gemini_history)
            with st.chat_message(
                name=MODEL_ROLE,
                avatar=AI_AVATAR_ICON,
            ):
                st.markdown(response)

            st.session_state.messages.append(
                dict(
                    role=MODEL_ROLE,
                    content=response,
                    avatar=AI_AVATAR_ICON,
                )
            )

            st.session_state.gemini_history.append({"role": "user", "parts": [{"text": prompt}]})
            st.session_state.gemini_history.append({"role": MODEL_ROLE, "parts": [{"text": response}]})

            # Save chat history
            joblib.dump(
                st.session_state.messages,
                f'{DATA_DIR}/{st.session_state.chat_id}-st_messages',
            )
            joblib.dump(
                st.session_state.gemini_history,
                f'{DATA_DIR}/{st.session_state.chat_id}-gemini_messages',
            )

        # Sidebar for document processing
        with st.sidebar:
            st.write("# Code Compliance:")
            url = st.text_area("Enter your Github Url here:")
            code = get_github_repo_contents(url)[:1000]
            print("code type", code)

            st.header("Compliance Regulations")
            if st.button("HIPA"):
                st.write("Optimizing for HIPAA Answers...")
            if st.button("GDPR"):
                st.write("Optimizing for General Data Protection Regulation (GDPR) Answers...")
            if st.button("EU AI Act"):
                st.write("Optimizing for EU AI Act compliance Answers...")


            if st.button("Submit & Process Code"):
                with st.spinner("Processing Code..."):
                    context = code

                    compliance_report = check_code_compliance(context)

                    with st.chat_message(
                        name=MODEL_ROLE,
                        avatar=AI_AVATAR_ICON,
                    ):
                        st.markdown(compliance_report)

                    st.session_state.messages.append(
                        dict(
                            role=MODEL_ROLE,
                            content=compliance_report,
                            avatar=AI_AVATAR_ICON,
                        )
                    )

                    st.session_state.gemini_history.append({"role": "user", "parts": [{"text": context}]})
                    st.session_state.gemini_history.append({"role": MODEL_ROLE, "parts": [{"text": compliance_report}]})

                    # Save chat history after processing the code input
                    joblib.dump(
                        st.session_state.messages,
                        f'{DATA_DIR}/{st.session_state.chat_id}-st_messages',
                    )
                    joblib.dump(
                        st.session_state.gemini_history,
                        f'{DATA_DIR}/{st.session_state.chat_id}-gemini_messages',
                    )
                    st.success("Code processed")
    elif page == "Questionnaire":
        # The pickle files saved will have the information, query into the pickle file to get the responses from the user
        response_code_object = questionnaire()

        # Load responses and run compliance check if responses exist
        if response_code_object != None:
            print(response_code_object) #passes in a dictionary
            project_compliance_checker = CodeProject(response_code_object)
            pia_question = "Please give any generated, observed, derived or inferred data processed by this project about a user?"
            answer_pia_question = answer_pia_questions_individually(project_compliance_checker, pia_question)
            print(answer_pia_question)
            st.write("**Compliance Report:**")
            st.markdown(compliance_report)



if __name__ == "__main__":
    main()
