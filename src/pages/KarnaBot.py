import os
import joblib
import streamlit as st
import requests
from dotenv import load_dotenv
import google.generativeai as genai

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.user_interface import user_input
from helper import *
import time
import hashlib
import random

# Load environment variables
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
    import fitz
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Main function for the KarnaBot page
def main():

    spinner_messages = ["Whats a lawyers favorite drink? Subpoeña coladas ", "Remember this isn't legal advice always consult a lawyer, like Better Call Saul", "How many lawyers does it take to screw a lightbulb? None they rested their case"]

    st.title("Welcome to KarnaBot")
    st.markdown("""
    KarnaBot is here to assist you with compliance checks, privacy impact assessments, and more.
    Use the chat interface below to interact with KarnaBot or fill out the questionnaire to get started.
    """)

    # Sidebar for past chats
    with st.sidebar:
        st.write('# KarnaBot')
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
        # Set the chat_id based on the prompt if it doesn't exist
        if st.session_state.get('chat_id') is None:
            st.session_state.chat_id = hashlib.sha256(prompt.encode()).hexdigest()
            st.session_state.chat_title = prompt

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

        # Show a spinner with alternating messages while generating the response
        # = ["Whats a lawyers favorite drink? Subpoeña coladas ", "Remember this isn't legal advice always consult a lawyer, like Better Call Saul", "Nick is an amazing professor"]
        with st.spinner(random.choice(spinner_messages)):
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

    # Sidebar for potential questions
    with st.sidebar:
        st.write("## Potential Questions")
        potential_questions = [
            "What is the difference between GDPR and the EU AI Act?",
            "What sensitive data is mentioned in GDPR?",
            "If I am building a machine learning model that uses information about a user, is that GDPR compliant?",
            "If I use a person’s last name within my analysis, is that in compliance with HIPAA?",
            "Can I collect users social security information of the california privacy act?",
        ]
        for question in potential_questions:
            if st.button(question):
                st.session_state['auto_filled'] = question

    # Automatically fill the chat input with a question from the sidebar
    if 'auto_filled' in st.session_state:
        prompt = st.session_state.pop('auto_filled')
        with st.chat_message('user'):
            st.markdown(prompt)
        st.session_state.messages.append(
            dict(
                role='user',
                content=prompt,
            )
        )

        # Show a spinner with alternating messages while generating the response
        #spinner_messages = ["Whats a lawyers favorite drink? Subpoeña coladas ", "Remember this isn't legal advice always consult a lawyer, like Better Call Saul", "Nick is an amazing professor"]
        with st.spinner(random.choice(spinner_messages)):
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

if __name__ == "__main__":
    main()