import streamlit as st
import webbrowser
import re
from streamlit_image_select import image_select
from streamlit_extras.stylable_container import stylable_container
from streamlit_option_menu import option_menu


# Create a menu bar with a logo and simple navigation buttons
def create_menu_bar():
    with st.container():
        col1, col2, right_spacer = st.columns([1, 5, 1])  # Adjust column widths as needed

        # Display the logo
        with col1:
            with stylable_container(
            key="logo",
            css_styles="""
                img {
                display: block;
                margin-left: -100px;
                width: 200px;
                height: 50px;
                margin-top: -10px;
                border-radius: 6px; /* Adjust this value for desired roundness */
                border: 2px solid grey; /* Example: 3px solid black border */
             }"""):
                st.image('logo.png', width=120) 
            # st.image("logo.png", width=50)

        # Menu bar with simple navigation text
        with col2:
           selected = option_menu(
                menu_title=None,  # No title for the menu
                options=["Home", "About", "How It Works", "Services"],  # Menu options
                icons=[".", ".", ".", "."],  # Optional icons
                menu_icon="menu",  # Icon for the entire menu
                default_index=0,  # Default selected item
                orientation="horizontal",  # Horizontal layout
                styles={
                    "container": {"padding": "5px", "background-color": "white"},
                    "icon": {"color": "black", "font-size": "16px"},
                    "nav-link": {
                        "font-size": "12px",
                        "text-align": "center",
                        "margin": "0px",  # Add horizontal margin for spacing
                        "padding": "0px",
                        "font-weight": "normal",  # Ensure normal font weight
                    },
                    "nav-link-selected": {
                        "background-color": "lightgray",
                        "font-weight": "normal",  # Ensure normal font weight when selected
                    },
                },
            )

# Function to create GitHub repository section with clickable images
def create_github_repositories():
    with stylable_container(
            key="text1",
            css_styles="""
                button {
                display: block;
                margin-left: -70px;
                width: 200px;
                height: 40px;
                border-radius: 3px; /* Adjust this value for desired roundness */
                border: 2px solid light-grey; /* Example: 3px solid black border */
                font-weight:bold;
     }"""):   
        st.button("**GitHub Repositories**")

    # Define paths to GitHub repository images
    repo_1_img = "code_logo_1.png"  # Change to actual image paths
    repo_2_img = "code_logo_2.png"
    repo_3_img = "code_logo_3.png"

    repo_images = [repo_1_img, repo_2_img, repo_3_img]
    repo_names = ["Repo 1", "Repo 2", "Repo 3"]

    # Initialize session state for GitHub repository selection
    if "selected_repo" not in st.session_state:
        st.session_state.selected_repo = None

    # Allow the user to select a GitHub repository via clickable images
    selected_repo = image_select("", repo_images)

    # If a repository is selected, update session state
    if selected_repo:
        image_num = int(re.findall('\d+', selected_repo)[0])
        st.session_state.selected_repo = repo_names[image_num - 1]

    # Create 3 columns for the repositories with "Goto" buttons
    repo_urls = ["https://github.com/repo1", "https://github.com/repo2", "https://github.com/repo3"]

    col1, col2, col3 = st.columns(3)

    with col1:
        with stylable_container(
            key="goto1",
            css_styles="""
                button {
                display: block;
                margin-left: 0px;
                width: 230px;
                border-radius: 3px; /* Adjust this value for desired roundness */
                border: 2px solid light-grey; /* Example: 3px solid black border */
            }"""):   
            if st.button("Goto Repo 1"):
                webbrowser.open_new_tab(repo_urls[0])

    with col2:
        with stylable_container(
            key="goto2",
            css_styles="""
                button {
                display: block;
                margin-left: 0px;
                width: 230px;
                border-radius: 3px; /* Adjust this value for desired roundness */
                border: 2px solid light-grey; /* Example: 3px solid black border */
            }"""):   
            if st.button("Goto Repo 2"):
                webbrowser.open_new_tab(repo_urls[1])

    with col3:
        with stylable_container(
            key="goto3",
            css_styles="""
                button {
                display: block;
                margin-left: 0px;
                width: 230px;
                border-radius: 3px; /* Adjust this value for desired roundness */
                border: 2px solid light-grey; /* Example: 3px solid black border */
            }"""):   
            if st.button("Goto Repo 3"):
                webbrowser.open_new_tab(repo_urls[2])

# Function to create Compliance Documents section with clickable buttons
def create_compliance_documents():
    with stylable_container(
            key="text1",
            css_styles="""
                button {
                display: block;
                margin-left: -70px;
                width: 200px;
                height: 40px;
                border-radius: 3px; /* Adjust this value for desired roundness */
                border: 2px solid light-grey; /* Example: 3px solid black border */
                font-weight:bold;
     }"""):   
        st.button("**Compliance Documents**")

    # Define paths to GitHub repository images
    cpra_img = "cpra.png"  # Change to actual image paths
    gdpr_img = "gdpr.png"
    hipaa_img = "hipaa.png"

    compliance_images = [cpra_img, gdpr_img, hipaa_img]
    compliance_docs = ["California Privacy Rights Act", "GDPR", "HIPAA"]

    # Ensure session state for selected compliance documents
    if "selected_doc" not in st.session_state:
        st.session_state.selected_doc = None

    # Allow the user to select a GitHub repository via clickable images
    selected_doc = image_select("", compliance_images)

    # If a repository is selected, update session state
    i = 0
    if selected_doc:
        if "cpra" in selected_doc:
            i = 0
        elif "gdpr" in selected_doc:
            i = 1
        else:
            i = 2
        st.session_state.selected_doc = compliance_docs[i]

    # Create 3 columns for compliance document selections
    col1, col2, col3 = st.columns(3)

    # Clickable buttons for compliance documents
    with col1:
        with stylable_container(
                key="selected_document1",
                css_styles="""
                    button {
                    display: block;
                    margin-left: 0px;
                    width: 230px;
                    border-radius: 3px; /* Adjust this value for desired roundness */
                    border: 2px solid light-grey; /* Example: 3px solid black border */
                }"""):   
            if st.button("Select Documentation"):
                st.session_state.page = 'compliance_overview'
                st.experimental_rerun()

    with col2:
        with stylable_container(
            key="selected_document2",
            css_styles="""
                button {
                display: block;
                margin-left: 0px;
                width: 230px;
                border-radius: 3px; /* Adjust this value for desired roundness */
                border: 2px solid light-grey; /* Example: 3px solid black border */
            }"""):   
            if st.button("Select Documentation", key="selected_document_2"):
                st.session_state.page = 'compliance_overview'
                st.experimental_rerun()

    with col3:
        with stylable_container(
            key="selected_document3",
            css_styles="""
                button {
                display: block;
                margin-left: 0px;
                width: 230px;
                border-radius: 3px; /* Adjust this value for desired roundness */
                border: 2px solid light-grey; /* Example: 3px solid black border */
            }"""):   
            if st.button("Select Documentation", key="selected_document_3"):
                st.session_state.page = 'compliance_overview'
                st.experimental_rerun()

# Function to check for compliance after mandatory selections
def check_for_compliance():
    # Ensure both a GitHub repository and a compliance document are selected before proceeding
    if st.session_state.selected_repo and st.session_state.selected_doc:
        # Create a unique identifier based on the selected repository and compliance document
        identifier = f"{st.session_state.selected_repo}-{st.session_state.selected_doc}"
        
        with stylable_container(
            key="check_compliance",
            css_styles="""
                button {
                display: block;
                margin-left: 240px;
                margin-top: 30px;
                width: 230px;
                border-radius: 3px; /* Adjust this value for desired roundness */
                border: 2px solid light-grey; /* Example: 3px solid black border */
                background-color: #FFE8A3;
                }"""):
            if st.button("Check for Compliance!"):
                # st.success("Checking for compliance...")

                # This is where you can pass the identifier to the next step or page
                st.write(f"Identifier: {identifier}")
                
        with stylable_container(
            key="talk_to_karnabot",
            css_styles="""
                button {
                display: block;
                margin-left: 240px;
                margin-top: 10px;
                width: 230px;
                border-radius: 3px; /* Adjust this value for desired roundness */
                border: 2px solid light-grey; /* Example: 3px solid black border */
                background-color: #b0c4de;
                }"""):
            if st.button("Ask the KarnaBot!"):
                st.session_state.page = 'karnabot'
                st.experimental_rerun()
    else:
        st.warning("Select both a GitHub repository and a compliance document to proceed.")

# Main Streamlit page
def app():
    create_menu_bar()  # Create the menu bar
    create_github_repositories()  # Display GitHub repositories with clickable images and Goto buttons
    create_compliance_documents()  # Compliance documents with clickable cards
    check_for_compliance()  # Compliance check section

# if __name__ == "__main__":
#     app()
