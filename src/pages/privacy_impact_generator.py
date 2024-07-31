import streamlit as st
import os
import joblib
import requests
from dotenv import load_dotenv
import google.generativeai as genai
import time

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.pia_generator import *
from src.code_class import *
import nbformat

def extract_code_cells(notebook_content):
    """Extracts only the code cells from a Jupyter notebook, excluding those with multimedia content."""
    notebook = nbformat.reads(notebook_content, as_version=4)
    
    code_cells = []
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            # Check if the cell contains multimedia content
            if not any(output.get('data', {}).get('image/png') or output.get('data', {}).get('image/jpeg') for output in cell.get('outputs', [])):
                code_cells.append(cell)
    
    new_notebook = nbformat.v4.new_notebook(cells=code_cells, metadata=notebook.get("metadata", {}))
    
    return nbformat.writes(new_notebook)

def fetch_contents(url, code_file_extensions):
    """Fetches the contents of a GitHub repository, including processing Jupyter notebooks."""
    response = requests.get(url)
    response.raise_for_status()
    contents = response.json()
    
    code = ""
    
    for item in contents:
        if item['type'] == 'file' and any(item['name'].endswith(ext) for ext in code_file_extensions):
            file_response = requests.get(item['download_url'])
            file_response.raise_for_status()
            file_content = file_response.text
            
            if item['name'].endswith('.ipynb'):
                # Extract code cells if the file is a Jupyter notebook
                file_content = extract_code_cells(file_content)
            
            code += file_content + "\n"
        elif item['type'] == 'dir':
            code += fetch_contents(item['url'], code_file_extensions)
    
    return code

def get_github_repo_contents(repo_url):
    """Gets the contents of a GitHub repository."""
    # Extract owner and repo name from the URL
    repo_parts = repo_url.strip().split('/')
    owner = repo_parts[-2]
    repo = repo_parts[-1]

    # GitHub API URL for the repository contents
    api_url = f'https://api.github.com/repos/{owner}/{repo}/contents'
    
    # Define file extensions to consider as code files
    code_file_extensions = ['.ipynb', '.py', '.js', '.java', '.cpp', '.c', '.h', '.html', '.css']

    return fetch_contents(api_url, code_file_extensions)

def main():
    st.title("Privacy Impact Assessment")
    st.write("""Generate a PIA in less than 5 minutes!
             """)
    st.sidebar.title("Input Options")
    st.sidebar.write("Enter a GitHub URL or directly input your code below:")
    
    url = st.sidebar.text_input("Enter GitHub URL:")
    code_input = st.sidebar.text_area("Or input your code here:")

    code = ""
    if url:
        code = get_github_repo_contents(url)
        if code:
            st.sidebar.write("Repository information retrieved successfully.")
        else:
            st.sidebar.warning("Failed to retrieve repository information. Please check the URL and try again.")
    
    if code_input:
        code = code_input
    
    if st.sidebar.button("Generate Privacy Impact Assessment"):
        if code:
            with st.spinner("Did you know it costs a company on average $3,000 to create a PIA"):
                time.sleep(5)
                pia_output = generate_pia(code)
                st.write(pia_output)
        else:
            st.sidebar.warning("Please enter a valid GitHub URL or input your code directly.")

main()
