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

# Main function
def main():
    st.set_page_config("KarnaBot")
    st.header("KarnaBot")

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
        # """
        #        try:
        #     py_files = st.file_uploader(
        #         "Upload your Python (.py) files and Submit", type=["py"], accept_multiple_files=False
        #     )
        #     if st.button("Submit & Process Python Files"):
        #         with st.spinner("Processing Python File..."):
        #             if py_files:
        #                 raw_text = py_files.read().decode('utf-8')  # Read and decode the Python file
        #                 context = raw_text

        #                 compliance_chain = get_code_compliance()
        #                 compliance_result = compliance_chain({"context": context, "question": "Is this code compliant with GDPR and the EU AI Act?"})
        #                 response_text = compliance_result['output_text']

        #                 with st.chat_message(
        #                     name=MODEL_ROLE,
        #                     avatar=AI_AVATAR_ICON,
        #                 ):
        #                     st.markdown(response_text)

        #                 st.session_state.messages.append(
        #                     dict(
        #                         role=MODEL_ROLE,
        #                         content=response_text,
        #                         avatar=AI_AVATAR_ICON,
        #                     )
        #                 )

        #                 st.session_state.gemini_history.append({"role": "user", "parts": [{"text": raw_text}]})
        #                 st.session_state.gemini_history.append({"role": MODEL_ROLE, "parts": [{"text": response_text}]})

        #             # Save chat history after processing the Python file
        #             joblib.dump(
        #                 st.session_state.messages,
        #                 f'{DATA_DIR}/{st.session_state.chat_id}-st_messages',
        #             )
        #             joblib.dump(
        #                 st.session_state.gemini_history,
        #                 f'{DATA_DIR}/{st.session_state.chat_id}-gemini_messages',
        #             )
        #             st.success("Python file processed")
        # except Exception as e:
        #     st.warning("Please upload a Python file.") """

        st.markdown("---")
        code_input = st.text_area("Enter your Python code here:")
        if st.button("Submit & Process Code"):
            with st.spinner("Processing Code..."):
                context = code_input

                compliance_report = check_code_compliance(context)
                #compliance_result = compliance_chain({"context": context, "question": "Is this code compliant with GDPR and the EU AI Act?"})
                #response_text = compliance_report['output_text']

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

if __name__ == "__main__":
    main()
