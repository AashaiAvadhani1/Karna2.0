from typing import List
import os

import streamlit as st
from langchain.vectorstores import FAISS
#from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from src.vector_store import get_conversational_chain


def user_input(user_question: str, history: List[dict]):
    embeddings4 = HuggingFaceEmbeddings(model_name='LaBSE')

    faiss_index_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'faiss_index')
    new_db = FAISS.load_local(faiss_index_dir, embeddings4, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question, "history": history},
        return_only_outputs=True
    )

    return response["output_text"]
