import requests
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
