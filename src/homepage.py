import streamlit as st
from streamlit_option_menu import option_menu
import importlib.util
import os


st.set_page_config(
    page_title="Multipage App",
    page_icon="🖐️",
)
# Function to dynamically import a module
def load_page(page_name):
    page_path = f"pages/{page_name}.py"
    spec = importlib.util.spec_from_file_location(page_name, page_path)
    page = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(page)
    return page

# Get list of pages from the pages folder
pages_path = "pages"
page_files = [f[:-3] for f in os.listdir(pages_path) if f.endswith(".py")]
page_files = sorted(page_files)  # Optional: sort the pages alphabetically

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=page_files,
        icons=["file"] * len(page_files),  # Using a generic icon for all pages
        menu_icon="cast",
        default_index=0,
    )


# Load the selected page
try:
    page = load_page(selected)
    page.main()  # Each page should have a main function
except Exception as e:
    st.error(f"Error loading page '{selected}': {e}")

# Main Page content (Optional: remove if not needed)
if selected == page_files[0]:  # If the first page is selected (can be changed to any condition)
    st.title("Main Page")

    # Overview metrics
    st.header("App Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Compliance Checks Done", value=123)  # Replace with actual value
    with col2:
        st.metric(label="Compliance Issues Faced", value=10)  # Replace with actual value
    with col3:
        st.metric(label="Potential Compliance Risks", value=5)  # Replace with actual value
    
    # Optional user input
    if "my_input" not in st.session_state:
        st.session_state["my_input"] = ""

    my_input = st.text_input("Input a text here", st.session_state["my_input"])
    submit = st.button("Submit")
    if submit:
        st.session_state["my_input"] = my_input
        st.write("You have entered: ", my_input)
