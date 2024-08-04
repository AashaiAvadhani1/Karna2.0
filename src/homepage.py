import streamlit as st
from streamlit_option_menu import option_menu
import importlib.util
import os
import pandas as pd

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

with st.sidebar:
    # Get list of pages from the pages folder
    pages_path = "pages"
    page_files = [f[:-3] for f in os.listdir(pages_path) if f.endswith(".py")]
    page_files = sorted(page_files)  # Optional: sort the pages alphabetically

# Main Page content (Optional: remove if not needed)
st.title("Main Page")

# Overview metrics
st.header("App Overview")

# Create a panel like the one in the screenshot
col1, col2, col3, col4 = st.columns(4)

# Metrics that update dynamically
# These values should be updated dynamically from your data source
total_assessments = 38  # Replace with dynamic value
my_assessments = 0  # Replace with dynamic value
total_dsrs = 83  # Replace with dynamic value
my_dsrs = 0  # Replace with dynamic value

with col1:
    st.metric(label="Total PIAs Completed", value=total_assessments)

with col2:
    st.metric(label="Current EU AI Act Violations", value=my_assessments)

with col3:
    st.metric(label="Current GDPR Violations", value=total_dsrs)

with col4:
    st.metric(label="Current GDPR Violations", value=my_dsrs)

# Optional user input
if "my_input" not in st.session_state:
    st.session_state["my_input"] = ""

#my_input = st.text_input("Input a text here", st.session_state["my_input"])
#submit = st.button("Submit")
#if submit:
#    st.session_state["my_input"] = my_input
#    st.write("You have entered: ", my_input)

# Compliance issues overview
st.header("Compliance Issues Overview")

# Sample data
compliance_data = pd.DataFrame({
    'Legislation': ['EU AI Act', 'GDPR', 'CCPA', 'HIPAA'],
    'Potential Concerns': [15, 30, 22, 10]
})

st.table(compliance_data)