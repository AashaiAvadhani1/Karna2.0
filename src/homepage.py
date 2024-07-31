import streamlit as st
from streamlit_option_menu import option_menu
import importlib.util
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

my_input = st.text_input("Input a text here", st.session_state["my_input"])
submit = st.button("Submit")
if submit:
    st.session_state["my_input"] = my_input
    st.write("You have entered: ", my_input)

# Graphs
st.header("Compliance Issues Related Graphs Overview")

# Sample data
data = pd.DataFrame({
    'Date': pd.date_range(start='1/1/2022', periods=100),
    'Value': np.random.randn(100).cumsum()
})

bar_data = pd.DataFrame({
    'Category': ['A', 'B', 'C', 'D'],
    'Values': np.random.rand(4) * 100
})

pie_data = pd.DataFrame({
    'Category': ['A', 'B', 'C', 'D'],
    'Values': np.random.rand(4) * 100
})

area_data = pd.DataFrame({
    'Date': pd.date_range(start='1/1/2022', periods=100),
    'Value': np.random.randn(100).cumsum()
})

# Displaying graphs in a 4x4 grid
col1, col2 = st.columns(2)

with col1:
    st.subheader("Compliance Issues - Line Chart")
    st.line_chart(data.set_index('Date'))

    st.subheader("Compliance Issues - Bar Chart")
    st.bar_chart(bar_data.set_index('Category'))

with col2:
    st.subheader("Compliance Issues - Pie Chart")
    fig, ax = plt.subplots()
    ax.pie(pie_data['Values'], labels=pie_data['Category'], autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

    st.subheader("Compliance Issues - Area Chart")
    st.area_chart(area_data.set_index('Date'))
