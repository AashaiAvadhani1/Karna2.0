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
import fitz  # PyMuPDF for PDF processing

import time as time

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
    st.title("Code Compliance Onboarding Questionnaire")
    
    # Example questions
    company_name = st.text_input("Company Name:")
    project_name = st.text_input("Project Name:")
    code_complexity = st.selectbox("Code Complexity", ["Simple", "Moderate", "Complex"])
    additional_notes = st.text_area("Additional Notes:")

    uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])
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
            "Code Complexity": code_complexity,
            "Additional Notes": additional_notes,
            "Uploaded File": uploaded_file.name if uploaded_file else None
        }
        joblib.dump(responses, os.path.join(DATA_DIR, 'questionnaire_responses.pkl'))

# Main function
def main():
    st.set_page_config("KarnaBot")
    st.header("KarnaBot")

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
            st.header("Upload Code Here:")
            st.markdown("---")
            code_input = st.text_area("Enter your Python code here:")
            if st.button("Submit & Process Code"):
                with st.spinner("Processing Code..."):
                    context = code_input

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
        #the pickle files saved will have the information, query into the pickle file to get the responses from the user
        questionnaire()

        # Load responses and run compliance check if responses exist
        responses = load_questionnaire_responses()
        if responses:
            compliance_report = develop_privacy_impact_assessment(
                responses["Company Name"],
                responses["Project Name"],
                responses["Code Complexity"],
                responses["Additional Notes"],
                responses["Uploaded File"]
            )
            st.write("**Compliance Report:**")
            st.markdown(compliance_report)
        
        #implement the privacy impact assessment legislation
        

if __name__ == "__main__":
    main()
